import copy as cp


import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer)
from mmcv.runner.hooks import Fp16OptimizerHook

from ..core import (DistEpochEvalHook, EpochEvalHook_MP, PairRunner,
                    OmniSourceDistSamplerSeedHook, OmniSourceRunner)
from ..datasets import build_dataloader, build_dataset
from ..utils import get_root_logger
import copy




def train_MP_model(#recognizer_model,
                model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """

    torch.cuda.empty_cache()

    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    assert not cfg.omnisource

    data_loaders = [
      build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]
    # put model on gpus
    assert not distributed
    #recognizer_model = MMDataParallel(
    #    recognizer_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    model = MMDataParallel(
        model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    # SGD(
    #     Parameter Group 0
    #     dampening: 0
    #     lr: 0.001
    #     momentum: 0.9
    #     nesterov: False
    #     weight_decay: 0.0005
    # )
    # <class 'torch.optim.sgd.SGD'>

    Runner = OmniSourceRunner if cfg.omnisource else PairRunner # EpochBasedRunner
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)


    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    # cfg.log_config['interval'] = 1
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # StepLrUpdate, GrandNormClip, Checkpoint, Log
    assert not distributed
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=15,#cfg.data.get('videos_per_gpu', 2),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 0),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=True,##False,
            drop_last=True)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEpochEvalHook if distributed else EpochEvalHook_MP
        eval_cfg['interval'] = 1 # interval=1
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    #if cfg.resume_from: # resume recognition model
    #    runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner_kwargs = dict()
    assert not cfg.omnisource
    # print(data_loaders)
    # print(cfg.workflow)
    # print(cfg.total_epochs)
    # [ < torch.utils.data.dataloader.DataLoader object at 0x7f6fe72c9690 >]
    # [('train', 1)]
    # 45

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, **runner_kwargs)
