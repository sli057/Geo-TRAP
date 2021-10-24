import argparse
import os
import os.path as osp
import time
import mmcv
import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmcv.runner import init_dist, set_random_seed, load_checkpoint
from mmcv.parallel import MMDataParallel
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import collect_env, get_root_logger
import numpy as np
from tools import ucf_cls2label, jester_cls2label
from query_attack import perturbation_image_decompose_from_scratch


def parse_args():
    parser = argparse.ArgumentParser(description='Decompose Attack')
    parser.add_argument(
        '--targeted',
        action='store_true',
        help='whether to do targeted attack')
    parser.add_argument('--work-dir', default=None,
                        help='the name of the work dir to save logs and models')
    parser.add_argument('--config',
                        default='configs/recognition/c3d/c3d_jester_MotionPerturbation.py',
                        # default='configs/recognition/c3d/c3d_ucf_MotionPerturbation.py',
                        help='config file path of the dataset')
    parser.add_argument('--max_p', default=10, type=int, help='the perturbation budget')
    # parser.add_argument('--random_seed', default=1.0, type=float, help='the perturbation budget')
    parser.add_argument('--transform_type_query', help='the name of the work dir to save logs and models')
    parser.add_argument('--config_rec',
        default='configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py',
        # for jester
        help='config file path of the recognizer')
    parser.add_argument(
        '--checkpoint_rec',
        default='work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth',
        #default='checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth',
        help='the checkpoint file to resume from for the recognizer')

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    # random.seed(args.random_seed)
    cfg = Config.fromfile(args.config)
    cfg_rec = Config.fromfile(args.config_rec)
    # print(args.cfg_options)
    # print(type(args.cfg_options))
    # {'optimizer.lr': 0.01}
    # <class 'dict'>
    cfg_rec.merge_from_dict(args.cfg_options)

    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set cudnn_benchmark
    if cfg_rec.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if cfg_rec.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg_rec.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg_rec.work_dir = osp.join(cfg_rec.work_dir, args.work_dir)
    assert args.checkpoint_rec is not None

    if args.gpu_ids is not None:
        cfg_rec.gpu_ids = args.gpu_ids
    else:
        cfg_rec.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.

    distributed = False

    # The flag is used to determine whether it is omnisource training
    cfg_rec.setdefault('omnisource', False)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg_rec.work_dir))
    # dump config
    cfg_rec.dump(osp.join(cfg_rec.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg_rec.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg_rec.seed = args.seed
    meta['seed'] = args.seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))



    recognizer_model = build_model(
        cfg_rec.model, train_cfg=cfg_rec.train_cfg, test_cfg=cfg_rec.test_cfg)
    load_checkpoint(recognizer_model, args.checkpoint_rec, map_location='cpu')
    logger.info(f'Load checkpoint for the recognition model, {args.checkpoint_rec}')


    dataset = build_dataset(cfg.data.val, dict(test_mode=False,  # cfg.data.test: num clip=10 instead of 1
                                               sample_by_class=False))



    torch.cuda.empty_cache()

    logger = get_root_logger(log_level=cfg.log_level)

    # put model on gpus
    assert not distributed

    recognizer_model = MMDataParallel(recognizer_model, device_ids=[0])
    recognizer_model.eval()

    success_list = []
    num_queries_list = []
    net_num_queries_list = []
    if 'jester' in args.config:
        label2cls = jester_cls2label
    elif 'ucf' in args.config:
        label2cls = ucf_cls2label
    else:
        print('Not sure which label2cls to use')
        exit()

    if 'jester' in args.config:
        label2cls = jester_cls2label
        cnt_per_cls = 2
    elif 'ucf' in args.config:
        label2cls = ucf_cls2label
        cnt_per_cls = 1
    else:
        print('Not sure which datatset is used')
        exit()
    all_cls = list(range(len(label2cls))) * cnt_per_cls
    # prog_bar = mmcv.ProgressBar(max_test)

    for data in dataset:
        if len(all_cls) == 0:
            break
        start = time.time()
        # prog_bar.update()
        img, flow, label = data[0]['imgs'], data[1]['imgs'], data[0]['label']
        if label.item() not in all_cls:
            continue
        with torch.no_grad():
            p_imgs = img.cuda()  # [1,3,16,112,112]
            if 'tsm' in args.config_rec:
                p_imgs = torch.transpose(p_imgs, 1, 2)
            else:
                p_imgs = torch.unsqueeze(p_imgs, 1)
            output = recognizer_model(p_imgs, label=None, return_loss=False)
        output_label = np.argmax(output[0])
        if output_label != label.item():
            continue
        # print(img.size(), torch.min(img), torch.max(img))
        # print(flow.size(), torch.min(flow), torch.max(flow))
        # print(label)
        # torch.Size([1, 3, 16, 112, 112]), tensor(-128.), tensor(122.)
        # torch.Size([1, 2, 16, 112, 112]), tensor(-1.), tensor(0.9922)
        # tensor([1])
        # cls_id = label[0]
        # print(img.size()) # torch.Size([1, 3, 16, 112, 112])
        feeddict = {'imgs1': img.cuda(), 'flow1': flow.cuda(), 'label1': label.cuda()}
        pred_adv_label, num_queries, success = perturbation_image_decompose_from_scratch(recognizer_model=recognizer_model,
                                                                                         feeddict=feeddict,
                                                                                         label2cls=label2cls,
                                                                                         config_rec=args.config_rec,
                                                                                         logger=logger,
                                                                                         code_p_func=code_p,
                                                                                         transform_type=args.transform_type_query,
                                                                                         targeted = args.targeted,
                                                                                         max_p=args.max_p)
        success_list.append(success)
        num_queries_list.append(num_queries)
        all_cls.remove(label.item())
        if success:
            net_num_queries_list.append(num_queries)
        logger.info('[{:d}/{:d}]: {:1.2f}s, {:d} out of {:d} attacks succeed, ANQ is {:1.2f}, ANQ of successfual attacks is {:1.2f}. '.format(
            len(success_list), len(label2cls)*cnt_per_cls, time.time()-start, np.sum(success_list), len(success_list),
            np.mean(num_queries_list), np.mean(net_num_queries_list)))


def code_p(p_static, p_motion):
    p = []
    p_static = p_static.squeeze(dim=-3) #[N,C,H,W]
    N, C, H, W = list(p_static.size())
    N, T, m1, m2 = list(p_motion.size())
    assert m1 == 3 and m2 == 3
    for frame_idx in range(T):
        motion = p_motion[:, frame_idx, :2, :] # [N,3,3] --> (N, 2, 3)
        align_corners = True
        flow_grid = nn.functional.affine_grid(theta=motion, size=p_static.size(),
                                              align_corners=align_corners)
        current_p_frame = nn.functional.grid_sample(input=p_static, grid=flow_grid, align_corners=align_corners)
        p.append(current_p_frame.unsqueeze(dim=-3))
    p = torch.cat(p, dim=-3)
    return p

if __name__ == '__main__':
    main()