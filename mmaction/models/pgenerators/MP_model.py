from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16


from mmaction.models import ResNet3d, ResNet3dSlowFast
from .resnet3d_dec import ResNetDec3d
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm
from mmaction.models import build_model
from mmcv import Config
from torchsummary import summary

class MPGeneratorTarget(nn.Module, metaclass=ABCMeta):
    pass


class MPGDTarget(nn.Module, metaclass=ABCMeta):
    pass


class MPGeneratorUnTargetStatic(nn.Module, metaclass=ABCMeta):
    pass

class MPGDUnTargetStatic(nn.Module, metaclass=ABCMeta):
    pass


class MPGeneratorUnTarget(nn.Module, metaclass=ABCMeta):
        """Class for MP Generator model.
            Args:
                input_flow (bool): whether the inputs are flow or RGB.
        """

        def __init__(self,
                     pretrained_encoder='checkpoints/slowfast_r50_video_4x16x1_256e_kinetics400_rgb_20200826-f85b90c5.pth',
                     resample_rate=1,
                     speed_ratio=1,
                     channel_ratio=1,
                     pathway=dict(
                         type='resnet3d',
                         depth=50,
                         pretrained=None,
                         lateral=False,
                         conv1_kernel=(5, 7, 7),
                         dilations=(1, 1, 1, 1),
                         conv1_stride_t=1,
                         pool1_stride_t=1,
                         inflate=(0, 0, 1, 1))):
            super().__init__()
            self.pretrained_encoder = pretrained_encoder

            self.encoder = ResNet3dSlowFast(pretrained=pretrained_encoder,
                                            dec_use=True,
                                            resample_rate=resample_rate,
                                            speed_ratio=speed_ratio,
                                            channel_ratio=channel_ratio,
                                            slow_pathway=pathway)
            self.decoder = ResNetDec3d(base_channels=64,
                                        depth=50,
                                        pretrained=None,
                                        static = False,
                                        num_stages=4,
                                        conv1_kernel=(1, 7, 7),
                                        dilations=(1, 1, 1, 1),
                                        conv1_stride_t=1,
                                        pool1_stride_t=1,
                                        inflate=(1, 1, 0, 0))

        def init_weights(self):
            self.encoder.init_weights()
            self.decoder.init_weights()

        def forward(self, x):
            # x1 = x1.reshape((-1,) + x1.shape[2:])
            # x2 = x2.reshape((-1,) + x2.shape[2:])

            x, _ = self.encoder(x)
            x = self.decoder(x)
            return x


class MPGDUnTarget(nn.Module, metaclass=ABCMeta):
        """Class for MP Generator model.
        Args:
            input_flow (bool): whether the inputs are flow or RGB.
        """

        def __init__(self,
                     recognizer,
                     input_flow=True,
                     norm_cfg=None,
                     init_std=0.005,
                     p_max=10,
                     class_level=False):
            super().__init__()
            self.input_flow = input_flow
            self.norm_cfg = norm_cfg
            self.init_std = init_std
            self.p_max = p_max
            self.class_level = class_level

            assert input_flow is False

            self.generator = MPGeneratorUnTarget()
            # pretrained recognizer
            self.recognizer = recognizer
            self.recognizer.eval()
            assert isinstance(recognizer, nn.Module)

        def train_step(self, data_batch, return_loss=True, **kwargs):
            """The iteration step during training.

            This method defines an iteration step during training, except for the
            back propagation and optimizer updating, which are done in an optimizer
            hook. Note that in some complicated cases or models, the whole process
            including back propagation and optimizer updating is also defined in
            this method, such as GAN.

            Args:
                data_batch (dict): The output of dataloader.
                optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                    runner is passed to ``train_step()``. This argument is unused
                    and reserved.

            Returns:
                dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                    ``num_samples``.
                    ``loss`` is a tensor for back propagation, which can be a
                    weighted sum of multiple losses.
                    ``log_vars`` contains all the variables to be sent to the
                    logger.
                    ``num_samples`` indicates the batch size (when the model is
                    DDP, it means the batch size on each GPU), which is used for
                    averaging the logs.
            """
            outputs = self(data_batch, return_loss, **kwargs)
            # for key in outputs:
            #     print(key, outputs[key])
            # top1_acc tensor(0., device='cuda:0', dtype=torch.float64)
            # top5_acc tensor(0.0667, device='cuda:0', dtype=torch.float64)
            # loss_cls tensor(20.0866, device='cuda:0', grad_fn=<MulBackward0>)
            loss, log_vars = self.recognizer._parse_losses(outputs)
            # print(loss)
            # print(log_vars)
            # tensor(0.0104, device='cuda:0', grad_fn= < AddBackward0 >)
            # OrderedDict([('top1_acc', 1.0), ('top5_acc', 1.0), ('loss_cls', 0.010394270531833172),
            #              ('loss', 0.010394270531833172)]
            loss = -loss
            log_vars['loss_cls'] = -log_vars['loss_cls']
            log_vars['loss'] = -log_vars['loss']

            output = dict(
                loss= loss,
                log_vars=log_vars,
                num_samples=len(next(iter(data_batch.values())))
            )
            return output

        def val_step(self, data_batch, return_loss=True, output_p=False, **kwargs):
            """The iteration step during validation.

            This method shares the same signature as :func:`train_step`, but used
            during val epochs. Note that the evaluation after training epochs is
            not implemented with this method, but an evaluation hook.
            """
            if output_p:
                return self(data_batch, return_loss, output_p, **kwargs)
            outputs = self(data_batch, return_loss, **kwargs)
            loss, log_vars = self.recognizer._parse_losses(outputs)

            loss = -loss
            log_vars['loss_cls'] = -log_vars['loss_cls']
            log_vars['loss'] = -log_vars['loss']

            output = dict(
                loss= loss,
                log_vars=log_vars,
                num_samples=len(next(iter(data_batch.values())))
            )
            return output

        def forward(self, data_batch, return_loss=False, output_p=False, **kwargs):
            if self.input_flow:
                x = data_batch['flow1']*128
            else:
                x = data_batch['imgs1']
            # label1, label2 = data_batch['label1'], data_batch['label2']
            imgs = data_batch['imgs1']
            label = data_batch['label1']
            p = self.generator(x)
            #print(p.size())
            if self.class_level: #True:#
                p1 = p[torch.randperm(p.size()[0])]
                #assert torch.any(p != p1)
                p = p1
            #print(p.size())
            p_imgs = imgs + p * self.p_max
            aux_info = {}
            if self.recognizer.cls_head.__class__.__name__ == 'TSMHead':
                p_imgs = torch.transpose(p_imgs, dim0=1, dim1=2)
            else:
                p_imgs = torch.unsqueeze(p_imgs, dim=1)
            self.recognizer.eval()
            # print(p_imgs.size())
            outputs = self.recognizer(p_imgs, label, return_loss=return_loss, **aux_info)
            if output_p:
                return outputs, p * self.p_max
            return outputs


if __name__ == '__main__':
    x1 = torch.tensor(np.random.rand(10, 3, 16, 112, 112), dtype=torch.float)
    generator = MPGeneratorUnTarget()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    summary(generator, (3, 16, 112, 112))
    #total parameter : 16M
    exit()

    p = generator(x1)
    print(p.size())
    exit()

    config_file = '../configs/recognition/c3d/c3d_ucf_MotionPerturbation.py'
    cfg = Config.fromfile(config_file)
    recognizer_model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    model = MPGeneratorTarget(recognizer_model)
    print(model)
    # x1 = torch.tensor(np.random.rand(10, 2, 16, 112, 112), dtype=torch.float) # batch_size, channel, frames, H, W
    # x2 = torch.tensor(np.random.rand(10, 2, 16, 112, 112), dtype=torch.float)
    # y = model(x1, x2)
    # print(np.shape(y)) #[10，3，16，112，112]
    # if hasattr(model, 'module'):
    #     model_name = model.module.__class__.__name__
    # else:
    #     model_name = model.__class__.__name__
    # print(model_name)

    # data_batch = dict()
    # #imgs = imgs.reshape((-1,) + imgs.shape[2:])
    # data_batch['imgs1'] = torch.tensor(np.random.rand(10, 3, 16, 112, 112), dtype=torch.float)
    # data_batch['imgs2'] = torch.tensor(np.random.rand(10, 3, 16, 112, 112), dtype=torch.float)
    # data_batch['flow1'] = torch.tensor(np.random.rand(10, 2, 16, 112, 112), dtype=torch.float)
    # data_batch['flow2'] = torch.tensor(np.random.rand(10, 2, 16, 112, 112), dtype=torch.float)
    # data_batch['label1'] = torch.tensor(np.random.randint(0,100,size=(10)))
    # data_batch['label2'] =  torch.tensor(np.random.randint(0,100,size=(10)))
    # output = model.train_step(data_batch, None)
    # for key in output:
    #     print(key, output[key])