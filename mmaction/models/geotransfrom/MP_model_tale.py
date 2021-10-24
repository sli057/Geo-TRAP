from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16



from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm
from mmaction.models import build_model
from mmcv import Config


class MPGeneratorTarget(nn.Module, metaclass=ABCMeta):
    """Class for MP Generator model.
        Args:
            input_flow (bool): whether the inputs are flow or RGB.
        """

    def __init__(self,
                 input_flow=True,
                 norm_cfg=None,
                 init_std=0.005,
                 ):
        super().__init__()
        self.input_flow = input_flow
        self.norm_cfg = norm_cfg
        self.init_std = init_std

        conv_param = dict(
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))

        convtrans_param = dict(
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            conv_cfg=dict(type='Conv3d'),  # ''deconv3d'),
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))

        final_convtrans_param = dict(
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            conv_cfg=dict(type='Conv3d'),  # ''deconv3d'),
            norm_cfg=None,
            act_cfg=dict(type='Tanh'))

        input_channel = 2 if self.input_flow else 3
        # [2/3, 16, 112, 112]
        self.conv1a = ConvModule(input_channel, 64, **conv_param)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True)
        self.pool1_indices = None
        # [ 64, 16, 56, 56]
        self.conv2a = ConvModule(64, 128, **conv_param)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True)
        self.pool2_indices = None
        # [ 128, 8, 28, 28]
        self.conv3a = ConvModule(128, 256, **conv_param)
        self.conv3b = ConvModule(256, 256, **conv_param)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True)
        self.pool3_indices = None
        # [ 256, 4, 14, 14]

        # [ 256, 4*2, 14, 14]
        self.conv_merge1a = ConvModule(256, 256, **conv_param)
        self.conv_merge1b = ConvModule(256, 256, **conv_param)
        self.pool_merge1 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        # [ 256, 4, 14, 14]
        self.unpool3 = nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.convtrans3b = ConvModule(256, 256, **convtrans_param)  # ConvModule(256, 256, **c3d_conv_param) #
        self.convtrans3a = ConvModule(256, 128, **convtrans_param)
        # [ 128, 8, 28, 28]
        self.unpool2 = nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.convtrans2a = ConvModule(128, 64, **convtrans_param)
        # [ 64, 16, 56, 56]
        self.unpool1 = nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.convtrans1a = ConvModule(64, 3, **final_convtrans_param)
        # [3, 16, 112, 112]

        # init_weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=self.init_std)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def encoder(self, x):
        x = self.conv1a(x)
        x, self.pool1_indices = self.pool1(x)

        x = self.conv2a(x)
        x, self.pool2_indices = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x, self.pool3_indices = self.pool3(x)

        return x

    def merge(self, x1, x2):
        x = torch.stack([x1, x2])  # [2, N, 256, 4, 14, 14]
        x = torch.transpose(x, 0, 1)  # [N, 2, 256, 4, 14, 14]
        x = torch.transpose(x, 0, -3)  # [4, 2, 256, N, 14, 14]
        x = torch.flatten(x, 0, 1)  # [8, 256, N, 14, 14]
        x = torch.transpose(x, 0, -3)  # [N, 256, 8, 14, 14]
        return x

    def decoder(self, x):
        x = self.conv_merge1a(x)
        x = self.conv_merge1b(x)
        x = self.pool_merge1(x)

        x = self.unpool3(x, self.pool3_indices)
        x = self.convtrans3b(x)
        x = self.convtrans3a(x)

        x = self.unpool2(x, self.pool2_indices)
        x = self.convtrans2a(x)

        x = self.unpool1(x, self.pool1_indices)
        x = self.convtrans1a(x)

        return x

    def forward(self, x1, x2):
        # x1 = x1.reshape((-1,) + x1.shape[2:])
        # x2 = x2.reshape((-1,) + x2.shape[2:])
        x1, x2 = self.encoder(x1), self.encoder(x2)  # [ N, 256, 4, 14, 14]
        x = self.merge(x1, x2)
        x = self.decoder(x)
        return x


class MPGDTarget(nn.Module, metaclass=ABCMeta):
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

        self.generator = MPGeneratorTarget(input_flow,  norm_cfg, init_std)
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
        output = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values())))
        )
        return output

    def val_step(self, data_batch, return_loss=True, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        outputs = self(data_batch, return_loss, **kwargs)
        loss, log_vars = self.recognizer._parse_losses(outputs)
        output = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples = len(next(iter(data_batch.values())))
        )
        return output

    def forward(self, data_batch, return_loss=False, **kwargs):
        if self.input_flow:
            x1, x2 = data_batch['flow1']*128, data_batch['flow2']*128
        else:
            x1, x2 = data_batch['imgs1'], data_batch['imgs2']
        #label1, label2 = data_batch['label1'], data_batch['label2']
        imgs = data_batch['imgs1']
        #print(torch.squeeze(data_batch['label2']))
        #print(torch.squeeze(data_batch['label1']))
        label = data_batch['label2']
        p = self.generator(x1, x2)
        if self.class_level:
            p = p[torch.randperm(p.size()[0])]
        p_imgs = imgs + p*self.p_max
        aux_info = {}
        outputs = self.recognizer(torch.unsqueeze(p_imgs,1), label, return_loss=return_loss, **aux_info)

        # label = data_batch['label1']
        # outputs = self.recognizer(torch.unsqueeze(imgs, 1), label, return_loss=return_loss, **aux_info)

        # print('shape, min, max of x1:', list(x1.size()), torch.min(x1), torch.max(x1))
        # print('shape, min, max of x2:', list(x2.size()), torch.min(x2), torch.max(x2))
        # print('shape, min, max of p:', list(p.size()), torch.min(p), torch.max(p))
        # print('shape, min, max of imgs:', list(imgs.size()), torch.min(imgs), torch.max(imgs))
        # print('shape of label:', list(label.size()))
        # print('label from:', torch.squeeze(data_batch['label1']))
        # print('label to:', torch.squeeze(label))

        # shape, min, max of x1: [15, 3, 16, 112, 112] tensor(-128., device='cuda:0') tensor(151., device='cuda:0')
        # shape, min, max of x2: [15, 3, 16, 112, 112] tensor(-128., device='cuda:0') tensor(151., device='cuda:0')
        # shape, min, max of imgs: [15, 3, 16, 112, 112] tensor(-128., device='cuda:0') tensor(151., device='cuda:0')
        # shape, min, max of p: [15, 3, 16, 112, 112] tensor(-0.1876, device='cuda:0', grad_fn= < MinBackward1 >)
        #                                               tensor(0.2187, device='cuda:0',grad_fn= < MaxBackward1 >)
        # shape of label: [15, 1]
        # label from: tensor([53, 43, 82, 41, 29, 77, 68, 74, 37, 21, 31, 77, 30, 28, 42], device='cuda:0')
        # label to: tensor([42, 63, 89, 51, 75, 64, 56, 61, 10, 26, 89, 26, 72, 100, 80], device='cuda:0')

        # print(torch.equal(x1,x2)) # False

        # shape, min, max of x1: [15, 2, 16, 112, 112] tensor(-0.9141, device='cuda:0') tensor(0.8516, device='cuda:0')
        # shape, min, max of x2: [15, 2, 16, 112, 112] tensor(-1., device='cuda:0') tensor(0.9531, device='cuda:0')

        return outputs


class MPGeneratorUnTarget(nn.Module, metaclass=ABCMeta):
        """Class for MP Generator model.
            Args:
                input_flow (bool): whether the inputs are flow or RGB.
        """

        def __init__(self,
                     input_flow=True,
                     norm_cfg=None,
                     init_std=0.005,
                     ):
            super().__init__()
            self.input_flow = input_flow
            self.norm_cfg = norm_cfg
            self.init_std = init_std

            conv_param = dict(
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='ReLU'))

            convtrans_param = dict(
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                conv_cfg=dict(type='Conv3d'),  # ''deconv3d'),
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='ReLU'))

            final_convtrans_param = dict(
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                conv_cfg=dict(type='Conv3d'),  # ''deconv3d'),
                norm_cfg=None,
                act_cfg=dict(type='Tanh'))

            input_channel = 2 if self.input_flow else 3
            # [2/3, 16, 112, 112]
            self.conv1a = ConvModule(input_channel, 64, **conv_param)
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True)
            self.pool1_indices = None
            # [ 64, 16, 56, 56]
            self.conv2a = ConvModule(64, 128, **conv_param)
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True)
            self.pool2_indices = None
            # [ 128, 8, 28, 28]
            self.conv3a = ConvModule(128, 256, **conv_param)
            self.conv3b = ConvModule(256, 256, **conv_param)
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True)
            self.pool3_indices = None
            # [ 256, 4, 14, 14]


            # [ 256, 4, 14, 14]
            self.unpool3 = nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.convtrans3b = ConvModule(256, 256, **convtrans_param)  # ConvModule(256, 256, **c3d_conv_param) #
            self.convtrans3a = ConvModule(256, 128, **convtrans_param)
            # [ 128, 8, 28, 28]
            self.unpool2 = nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.convtrans2a = ConvModule(128, 64, **convtrans_param)
            # [ 64, 16, 56, 56]
            self.unpool1 = nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.convtrans1a = ConvModule(64, 3, **final_convtrans_param)
            # [3, 16, 112, 112]

            # init_weights
            self.init_weights()

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=self.init_std)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

        def encoder(self, x):
            x = self.conv1a(x)
            x, self.pool1_indices = self.pool1(x)

            x = self.conv2a(x)
            x, self.pool2_indices = self.pool2(x)

            x = self.conv3a(x)
            x = self.conv3b(x)
            x, self.pool3_indices = self.pool3(x)

            return x

        def decoder(self, x):


            x = self.unpool3(x, self.pool3_indices)
            x = self.convtrans3b(x)
            x = self.convtrans3a(x)

            x = self.unpool2(x, self.pool2_indices)
            x = self.convtrans2a(x)

            x = self.unpool1(x, self.pool1_indices)
            x = self.convtrans1a(x)

            return x

        def forward(self, x):
            # x1 = x1.reshape((-1,) + x1.shape[2:])
            # x2 = x2.reshape((-1,) + x2.shape[2:])
            x = self.encoder(x)  # [ N, 256, 4, 14, 14]
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

            self.generator = MPGeneratorUnTarget(input_flow, norm_cfg, init_std)
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



class MPGeneratorUnTargetStatic(nn.Module, metaclass=ABCMeta):
        """Class for MP Generator model.
            Args:
                input_flow (bool): whether the inputs are flow or RGB.
        """

        def __init__(self,
                     input_flow=True,
                     norm_cfg=None,
                     init_std=0.005,
                     ):
            super().__init__()
            self.input_flow = input_flow
            self.norm_cfg = norm_cfg
            self.init_std = init_std

            conv_param = dict(
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='ReLU'))

            convtrans_param = dict(
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                conv_cfg=dict(type='Conv3d'),  # ''deconv3d'),
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type='ReLU'))

            final_convtrans_param = dict(
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                conv_cfg=dict(type='Conv3d'),  # ''deconv3d'),
                norm_cfg=None,
                act_cfg=dict(type='Tanh'))

            input_channel = 2 if self.input_flow else 3
            # [2/3, 16, 112, 112]
            self.conv1a = ConvModule(input_channel, 64, **conv_param)
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True)
            self.pool1_indices = None
            # [ 64, 16, 56, 56]
            self.conv2a = ConvModule(64, 128, **conv_param)
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True)
            self.pool2_indices = None
            # [ 128, 8, 28, 28]
            self.conv3a = ConvModule(128, 256, **conv_param)
            self.conv3b = ConvModule(256, 256, **conv_param)
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True)
            self.pool3_indices = None
            # [ 256, 4, 14, 14]


            # [ 256, 4, 14, 14]
            self.unpool3 = nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.convtrans3b = ConvModule(256, 256, **convtrans_param)  # ConvModule(256, 256, **c3d_conv_param) #
            self.convtrans3a = ConvModule(256, 128, **convtrans_param)
            # [ 128, 8, 28, 28]
            self.unpool2 = nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.convtrans2a = ConvModule(128, 64, **convtrans_param)
            # [ 64, 16, 56, 56]
            self.unpool1 = nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.convtrans1a = ConvModule(64, 3, **convtrans_param) # changed
            # [3, 16, 112, 112]

            # [3, 16, 112, 112]
            self.merge_pool1 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), return_indices=False)
            self.merge_conv1 = ConvModule(3, 3, **convtrans_param)
            # [3, 8, 112, 112]
            self.merge_pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), return_indices=False)
            self.merge_conv2 = ConvModule(3, 3, **convtrans_param)
            # [3, 4, 112, 112]
            self.merge_pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), return_indices=False)
            self.merge_conv3 = ConvModule(3, 3, **convtrans_param)
            # [3, 2, 112, 112]
            self.merge_pool4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), return_indices=False)
            self.merge_conv4 = ConvModule(3, 3, **final_convtrans_param)
            # [3, 1, 112, 112]

            # init_weights
            self.init_weights()

        def merge(self, x):
            x = self.merge_pool1(x)
            x = self.merge_conv1(x)
            x = self.merge_pool2(x)
            x = self.merge_conv2(x)
            x = self.merge_pool3(x)
            x = self.merge_conv3(x)
            x = self.merge_pool4(x)
            x = self.merge_conv4(x)
            return x




        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=self.init_std)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

        def encoder(self, x):
            x = self.conv1a(x)
            x, self.pool1_indices = self.pool1(x)

            x = self.conv2a(x)
            x, self.pool2_indices = self.pool2(x)

            x = self.conv3a(x)
            x = self.conv3b(x)
            x, self.pool3_indices = self.pool3(x)

            return x

        def decoder(self, x):
            x = self.unpool3(x, self.pool3_indices)
            x = self.convtrans3b(x)
            x = self.convtrans3a(x)

            x = self.unpool2(x, self.pool2_indices)
            x = self.convtrans2a(x)

            x = self.unpool1(x, self.pool1_indices)
            x = self.convtrans1a(x)

            return x

        def forward(self, x):
            # x1 = x1.reshape((-1,) + x1.shape[2:])
            # x2 = x2.reshape((-1,) + x2.shape[2:])
            x = self.encoder(x)  # [ N, 256, 4, 14, 14]
            x = self.decoder(x)
            x = self.merge(x)
            return x


class MPGDUnTargetStatic(nn.Module, metaclass=ABCMeta):
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

        self.generator = MPGeneratorUnTargetStatic(input_flow, norm_cfg, init_std)
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
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values())))
        )
        return output

    def val_step(self, data_batch, return_loss=True, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        outputs = self(data_batch, return_loss, **kwargs)
        loss, log_vars = self.recognizer._parse_losses(outputs)

        loss = -loss
        log_vars['loss_cls'] = -log_vars['loss_cls']
        log_vars['loss'] = -log_vars['loss']

        output = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values())))
        )
        return output

    def forward(self, data_batch, return_loss=False, **kwargs):
        if self.input_flow:
            x = data_batch['flow1'] * 128
        else:
            x = data_batch['imgs1']
        # label1, label2 = data_batch['label1'], data_batch['label2']
        imgs = data_batch['imgs1']
        label = data_batch['label1']
        p = self.generator(x)
        # [N, 3, 1, 112, 112]
        # print(p.size())
        p = p.repeat(1, 1, 16, 1, 1)
        # print(p.size())
        if self.class_level: #True: #for universal attack #
            p1 = p[torch.randperm(p.size()[0])]
            # assert torch.any(p != p1)
            p = p1
        # print(p.size())
        p_imgs = imgs + p * self.p_max
        aux_info = {}
        outputs = self.recognizer(torch.unsqueeze(p_imgs, 1), label, return_loss=return_loss, **aux_info)

        return outputs


if __name__ == '__main__':
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