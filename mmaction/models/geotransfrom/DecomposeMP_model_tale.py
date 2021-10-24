import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import _load_checkpoint, load_state_dict , load_checkpoint
from mmcv.utils import print_log

from ...utils import get_root_logger
from ..registry import BACKBONES
from mmaction.models import ResNet3d, ResNet3dSlowFast
from .resnet3d_dec import ResNetDec3d
from .regression_head import RegressionHead, RegressionHeadNoneCu
from abc import ABCMeta

import numpy as np
import matplotlib.pyplot as plt



class DecomposeG(nn.Module):
    """ Decompose based motion generation.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str): The file path to a pretrained model.
        resample_rate (int): A large temporal stride ``resample_rate``
            on input frames, corresponding to the :math:`\\tau` in the paper.
            i.e., it processes only one out of ``resample_rate`` frames.
            Default: 16.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Default: 8.
        slow_pathway (dict): Configuration of slow branch, should contain
            necessary arguments for building the specific type of pathway
            and:
            type (str): type of backbone the pathway bases on.
            lateral (bool): determine whether to build lateral connection
            for the pathway.Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=True, depth=50, pretrained=None,
                conv1_kernel=(1, 7, 7), dilations=(1, 1, 1, 1),
                conv1_stride_t=1, pool1_stride_t=1, inflate=(0, 0, 1, 1))

        fast_pathway (dict): Configuration of fast branch, similar to
            `slow_pathway`. Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=False, depth=50, pretrained=None, base_channels=8,
                conv1_kernel=(5, 7, 7), conv1_stride_t=1, pool1_stride_t=1)
    """

    def __init__(self,
                 pretrained_encoder=None,
                 pretrained=None,
                 use_resize=True,
                 resample_rate=8,
                 speed_ratio=8,
                 channel_ratio=8,
                 motion_scale=0.1,
                 regularization=False,
                 regularization_alpha=1,
                 slow_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     conv1_kernel=(1, 7, 7),
                     dilations=(1, 1, 1, 1),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     inflate=(0, 0, 1, 1)),
                 fast_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1)):
        super().__init__()
        self.pretrained = pretrained
        self.pretrained_encoder = pretrained_encoder
        self.motion_scale = motion_scale
        self.regularization = regularization
        self.regularization_alpha = regularization_alpha

        self.encoder = ResNet3dSlowFast(pretrained=pretrained_encoder,
                                        dec_use=True,
                                        resample_rate=resample_rate,
                                        speed_ratio=speed_ratio,
                                        channel_ratio=channel_ratio,
                                        slow_pathway=slow_pathway,
                                        fast_pathway=fast_pathway)
        self.slow_decoder = ResNetDec3d(base_channels=64,
                                        depth=50,
                                        pretrained=None,
                                        num_stages=4,
                                        conv1_kernel=(1, 7, 7),
                                        dilations=(1, 1, 1, 1),
                                        conv1_stride_t=1,
                                        pool1_stride_t=1,
                                        inflate=(1, 1, 0, 0))
        num_freedom = 4 if use_resize else 3
        self.fast_regression = RegressionHeadNoneCu(num_freedom=num_freedom,
                                              in_channels=8*32,
                                              use_resize=use_resize)
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            # Directly load 3D model.
            load_checkpoint(self, self.pretrained, strict=True, logger=logger, prefix='backbone')
        elif self.pretrained is None:
            # print(self.pretrained_encoder)
            # if self.pretrained_encoder is None:
            self.encoder.init_weights()
            # Init two branch seperately.
            self.slow_decoder.init_weights()
            self.fast_regression.init_weights()
            # self.slow_path.init_weights() # to be continued
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted
                by the backbone.
        """
        x_slow, x_fast = self.encoder(x)
        # print(x_slow.size(), x_fast.size())
        # x_slow [N, 64*16, 2, H/16=7, W/16=7]
        # x_fast [N, 8*32, 16, H/32=4, H/32=4]
        p_static = self.slow_decoder(x_slow)
        # print(torch.min(p_static), torch.max(p_static))
        # p_static [N, 3, 1, H=112, W=112]
        regularization_loss = torch.mean(torch.norm(self.fast_regression(x_fast), dim=-1))
        p_motion = self.fast_regression(x_fast) * self.motion_scale  #/10.0
        # print(p_motion[0]/self.motion_scale, regularization_loss, self.motion_scale)
        # print(p_static.size(), p_motion.size())
        # p_motion [N, 16, num_freedom],
        p = self.code_p(p_static, p_motion)

        if self.regularization:
            return p, regularization_loss * self.regularization_alpha
        else:
            return p

    def code_p_copy(self, p_static, p_motion):
        p = []
        #p_static = p_static.squeeze(dim=-3)
        last_p_frame = p_static.squeeze(dim=-3)
        for frame_idx in range(16):
            motion = p_motion[:, frame_idx, :]
            dx, dy, r_mag, r_size = motion[:, 0], motion[:, 1], motion[:, 2]+1, motion[:, 3]+1  # range(-1,1)
            current_p_frame = self.code_p_frame(last_p_frame, dx, dy, r_mag, r_size)
            current_p_frame = torch.clamp(current_p_frame, -1.0, 1.0)
            p.append(current_p_frame.unsqueeze(dim=-3))
            last_p_frame = p_static #current_p_frame
        p = torch.cat(p, dim=-3)
        return p

    def code_p(self, p_static, p_motion):
        # print(p_static.size(), p_motion.size())
        # return
        p = []
        p_static = p_static.squeeze(dim=-3)
        # p_motion += torch.randn(size=p_motion.size())*0.002
        # last_p_frame = p_static.squeeze(dim=-3)
        # print(last_p_fr   ame.size()) #[N, C, H, W]
        prefix_dx, prefix_dy, prefix_r_mag, prefix_r_size = 0.0, 0.0, 1.0, 1.0
        # print(p_motion[0, :, 0] / self.motion_scale) # [N, 16, 4]
        # print(p_static.requires_grad, p_motion.requires_grad)
        last_frame = p_static
        for frame_idx in range(16):
            motion = p_motion[:, frame_idx, :]
            dx, dy, r_mag, r_size = motion[:, 0], motion[:, 1], motion[:, 2]+1, motion[:, 3]+1  # range(-1,1)
            # print(dx.requires_grad, dy.requires_grad, r_mag.requires_grad, r_size.requires_grad)
            # exit()
            prefix_dx = torch.clamp(dx, -1.0, 1.0)  # [N]+prefix_dx
            # print(prefix_dx)
            prefix_dy = torch.clamp(dy, -1.0, 1.0)  # +prefix_dy
            prefix_r_mag = r_mag   #* prefix_r_mag
            # print(motion[:, 1])  # [N]
            # print(motion[:, 0])
            prefix_r_size = r_size   #* prefix_r_size
            # current_p_frame = self.code_p_frame(last_p_frame, p_motion[:,frame_idx,:])
            # current_p_frame = self.code_p_frame(p_static, dx, dy, r_mag, r_size)
            current_p_frame = self.code_p_frame(last_frame, prefix_dx, prefix_dy, prefix_r_mag, prefix_r_size)
            current_p_frame = torch.clamp(current_p_frame, -1.0, 1.0)
            p.append(current_p_frame.unsqueeze(dim=-3))
            last_frame =  p_static #current_p_frame
            # last_p_frame = current_p_frame
        p = torch.cat(p, dim=-3)
        return p

    def create_grid_flow(self, flow, N, H, W):
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W, 1).repeat(N, 1, 1, 1)
        yy = yy.view(1, H, W, 1).repeat(N, 1, 1, 1)
        grid = torch.cat((xx, yy), 3).float()
        if flow.is_cuda:
            grid = grid.cuda()
        # print(grid.requires_grad)
        # exit()
        vgrid = grid + flow
        # print(vgrid[0,:,:,0])
        # print(vgrid[0,:,:,1])
        # scale grid to [-1,1]
        vgrid[:, :, :, 0] = 2.0*vgrid[:, :, :, 0].clone() / max(W-1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0*vgrid[:, :, :, 1].clone() / max(H-1, 1) - 1.0
        # print(vgrid[0, :, :, 0])
        # print(vgrid[0, :, :, 1])
        # print(torch.min(vgrid[:, :, :, 0]), torch.max(vgrid[:, :, :, 0]),torch.min(vgrid[:, :, :, 1]), torch.max(vgrid[:, :, :, 1] ))
        return vgrid

    def create_grid_resize(self, scale_factor, N, H, W):
        # N, _, H, W = list(x.size())
        vgrid = []
        for batch_idx in range(N):
            xx = torch.arange(-(W - 1) / 2.0, (W - 1) / 2.0 + 1).view(1, -1).repeat(H, 1)
            yy = torch.arange(-(H - 1) / 2.0, (H - 1) / 2.0 + 1).view(-1, 1).repeat(1, W)
            # print(xx.requires_grad)
            if scale_factor.is_cuda:
                xx = xx.cuda()
                yy = yy.cuda()
            xx = xx * (1.0 / scale_factor[batch_idx]) + (W - 1) / 2.0
            # print(xx.requires_grad)
            # exit()
            yy = yy * (1.0 / scale_factor[batch_idx]) + (H - 1) / 2.0
            # print(xx)
            # print(yy)
            xx = xx.view(1, H, W, 1)  # .repeat(N, 1, 1, 1)
            yy = yy.view(1, H, W, 1)  # .repeat(N, 1, 1, 1)
            vgrid.append(torch.cat((xx, yy), 3).float())
        vgrid = torch.cat(vgrid, dim=0)
        # print(vgrid.size())

        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0].clone() / max(W - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1].clone() / max(H - 1, 1) - 1.0
        # print(vgrid[0, :, :, 0])
        # print(vgrid[0, :, :, 1])
        return vgrid

    def roll_xy(self, frame_batch, dx, dy):
        # print(frame_batch.size())
        N, C, H, W = list(frame_batch.size())
        dx = dx.to(dtype=torch.int)
        dy = dy.to(dtype=torch.int)
        #print(frame_batch.requires_grad, dx.requires_grad)
        # print(dx,dy)
        new_frame_batch = []
        for i in range(N):
            new_frame = torch.roll(frame_batch[i], (dx[i], dy[i]), dims=(-1, -2))
            # print(new_frame.requires_grad)
            new_frame_batch.append(new_frame)
            # print(new_frame)
        new_frame_batch = torch.stack(new_frame_batch, dim=0)
        # print(new_frame_batch.size())
        return new_frame_batch



    def code_p_frame(self, last_p_frame,  dx, dy, r_mag, r_size):
        # last_p_frame # [N, 3, H=112, W=112]
        # motion # [N, num_freedom]
        N, _,  H, W = list(last_p_frame.size())
        # dx, dy, m = motion[:, 0], motion[:, 1], motion[:, 2]  # range(-1,1)

        dx = (dx * W) #.to(torch.int) #[N]
        dy = (dy * H) #.to(torch.int) #[N]
        flow = torch.cat((dx.view(-1, 1), dy.view(-1, 1)), dim=1)  # alternative
        flow = flow.view(N, 1, 1, 2).repeat(1, H, W, 1) # flow size [N,H,W,2]  # alternative
        vgrid1 = self.create_grid_flow(flow, N, H, W)  # alternative
        # print(vgrid1.size())
        vgrid2 = self.create_grid_resize(r_size, N, H, W)
        # print(vgrid2.size())
        r_mag = r_mag.view((N, 1, 1, 1)).expand_as(last_p_frame) # range(0,2)
        #print(flow)
        #print(m)
        p_frame = nn.functional.grid_sample(last_p_frame*r_mag, vgrid1,
                                            mode='bilinear', padding_mode='zeros', align_corners=True)  # alternative
        # p_frame = self.roll_xy(last_p_frame*r_mag, dx, dy)
        p_frame = nn.functional.grid_sample(p_frame, vgrid2,
                                            mode='bilinear', padding_mode='zeros', align_corners=True)
        mask = torch.ones(p_frame.size())
        if r_size.is_cuda:
            mask = mask.cuda()
        # print('mask:', mask[0, 0, :, :])
        mask = nn.functional.grid_sample(mask, vgrid1, align_corners=True)  # alternative
        mask = nn.functional.grid_sample(mask, vgrid2, align_corners=True)
        # print('mask:', mask[0,0,:,:])
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        p_frame = p_frame * mask
        return p_frame

class DecomposeGDUnTarget(nn.Module, metaclass=ABCMeta):
    """Class for MP Generator model.
    Args:
        input_flow (bool): whether the inputs are flow or RGB.
    """

    def __init__(self,
                 recognizer,
                 use_resize=True,
                 pretrained_encoder=None,
                 pretrained=None,
                 regularization=False,
                 regularization_alpha=1,
                 motion_scale=0.1,
                 resample_rate=8,
                 speed_ratio=8,
                 channel_ratio=8,
                 p_max=10,
                 class_level=False):
        super().__init__()
        self.p_max = p_max
        self.class_level = class_level
        self.regularization = regularization
        if regularization:
            print('We are using regularization on the motion vector, alpha = {:1.1f}'.format(regularization_alpha))
        else:
            print('We are not using regularization')

        if use_resize:
            print('We are using resize parameter :)')
        else:
            print('We are not using resize parameter :(')

        self.generator = DecomposeG(pretrained_encoder=pretrained_encoder,
                                    pretrained=pretrained,
                                    use_resize=use_resize,
                                    regularization=regularization,
                                    regularization_alpha=regularization_alpha,
                                    motion_scale=motion_scale,
                                    resample_rate=resample_rate,
                                    speed_ratio=speed_ratio,
                                    channel_ratio=channel_ratio)
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
        if self.regularization:
            outputs, reg_loss = self(data_batch, return_loss, **kwargs)
        else:
            outputs = self(data_batch, return_loss, **kwargs)
        # for key in outputs:
        #     print(key, outputs[key])
        # top1_acc tensor(0., device='cuda:0', dtype=torch.float64)
        # top5_acc tensor(0.0667, device='cuda:0', dtype=torch.float64)
        # loss_cls tensor(20.0866, device='cuda:0', grad_fn=<MulBackward0>)
        loss, log_vars = self.recognizer._parse_losses(outputs)
        #assert loss == log_vars['loss']
        # print(loss)
        # print(log_vars)
        # tensor(0.0104, device='cuda:0', grad_fn= < AddBackward0 >)
        # OrderedDict([('top1_acc', 1.0), ('top5_acc', 1.0), ('loss_cls', 0.010394270531833172),
        #              ('loss', 0.010394270531833172)]
        #print(log_vars['loss'], reg_loss.item())
        #exit()
        if self.regularization:
            loss = -loss + reg_loss
            log_vars['loss_cls'] = -log_vars['loss_cls']
            log_vars['loss'] = -log_vars['loss'] + reg_loss.item()
        else:
            loss = -loss
            log_vars['loss_cls'] = -log_vars['loss_cls']
            log_vars['loss'] = -log_vars['loss']


        output = dict(
            loss=loss,
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

        if self.regularization:
            outputs, reg_loss = self(data_batch, return_loss, **kwargs)
        else:
            outputs = self(data_batch, return_loss, **kwargs)

        loss, log_vars = self.recognizer._parse_losses(outputs)

        if self.regularization:
            loss = -loss + reg_loss
            log_vars['loss_cls'] = -log_vars['loss_cls']
            log_vars['loss'] = -log_vars['loss'] + reg_loss.item()
        else:
            loss = -loss
            log_vars['loss_cls'] = -log_vars['loss_cls']
            log_vars['loss'] = -log_vars['loss']

        output = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values())))
        )
        return output

    def forward(self, data_batch, return_loss=False, output_p=False, **kwargs):
        imgs = data_batch['imgs1']
        label = data_batch['label1']
        if self.regularization:
            p, reg_loss = self.generator(imgs)
        else:
            p = self.generator(imgs)
        #print(torch.min(p), torch.max(p))
        # print(p.size())
        if self.class_level:  #
            p1 = p[torch.randperm(p.size()[0])]
            # assert torch.any(p != p1)
            p = p1
        # print(p.size())
        # print(p.size())

        # p = p[:, :, 0, :, :].unsqueeze(dim=-3).repeat([1, 1, 16, 1, 1])
        # print(p.size())
        p_imgs = imgs + p * self.p_max
        aux_info = {}
        outputs = self.recognizer(torch.unsqueeze(p_imgs, 1), label, return_loss=return_loss, **aux_info)
        if output_p:
            return outputs, p*self.p_max
        if self.regularization:
            return outputs, reg_loss

        return outputs


class DecomposeGDOneTarget(nn.Module, metaclass=ABCMeta):
    """Class for MP Generator model.
    Args:
        input_flow (bool): whether the inputs are flow or RGB.
    """

    def __init__(self,
                 recognizer,
                 target_label,
                 use_resize=True,
                 pretrained_encoder=None,
                 pretrained=None,
                 regularization=False,
                 regularization_alpha=1.0,
                 motion_scale=0.1,
                 resample_rate=8,
                 speed_ratio=8,
                 channel_ratio=8,
                 p_max=10,
                 class_level=False,
                 ):
        super().__init__()
        self.target_label = target_label
        self.p_max = p_max
        self.class_level = class_level
        self.regularization = regularization
        if regularization:
            print('We are using regularization on the motion vector, alpha = {:1.1f}'.format(regularization_alpha))
        else:
            print('We are not using regularization')

        if use_resize:
            print('We are using resize parameter :)')
        else:
            print('We are not using resize parameter :(')

        self.generator = DecomposeG(pretrained_encoder=pretrained_encoder,
                                    pretrained=pretrained,
                                    use_resize=use_resize,
                                    motion_scale=motion_scale,
                                    regularization=regularization,
                                    regularization_alpha=regularization_alpha,
                                    resample_rate=resample_rate,
                                    speed_ratio=speed_ratio,
                                    channel_ratio=channel_ratio)
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
        if self.regularization:
            outputs, reg_loss = self(data_batch, return_loss, **kwargs)
        else:
            outputs = self(data_batch, return_loss, **kwargs)
        # for key in outputs:
        #     print(key, outputs[key])
        # top1_acc tensor(0., device='cuda:0', dtype=torch.float64)
        # top5_acc tensor(0.0667, device='cuda:0', dtype=torch.float64)
        # loss_cls tensor(20.0866, device='cuda:0', grad_fn=<MulBackward0>)
        loss, log_vars = self.recognizer._parse_losses(outputs)

        if self.regularization:
            loss = loss + reg_loss
            log_vars['loss'] = log_vars['loss'] + reg_loss.item()

        output = dict(
            loss=loss,
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

        if self.regularization:
            outputs, reg_loss = self(data_batch, return_loss, **kwargs)
        else:
            outputs = self(data_batch, return_loss, **kwargs)
        loss, log_vars = self.recognizer._parse_losses(outputs)

        if self.regularization:
            loss = loss + reg_loss
            log_vars['loss'] = log_vars['loss'] + reg_loss.item()

        output = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values())))
        )
        return output

    def forward(self, data_batch, return_loss=False, output_p=False, **kwargs):
        imgs = data_batch['imgs1']
        label = torch.ones_like(data_batch['label1']) * self.target_label
        if self.regularization:
            p, reg_loss = self.generator(imgs)
        else:
            p = self.generator(imgs)
        #print(torch.min(p), torch.max(p))
        # print(p.size())
        if self.class_level:  #
            p1 = p[torch.randperm(p.size()[0])]
            # assert torch.any(p != p1)
            p = p1

        p_imgs = imgs + p * self.p_max
        aux_info = {}
        outputs = self.recognizer(torch.unsqueeze(p_imgs, 1), label, return_loss=return_loss, **aux_info)

        if output_p:
            return outputs, p*self.p_max
        if self.regularization:
            return outputs, reg_loss
        return outputs



class DecomposeTargetedG(nn.Module):
    """ Decompose based motion generation.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str): The file path to a pretrained model.
        resample_rate (int): A large temporal stride ``resample_rate``
            on input frames, corresponding to the :math:`\\tau` in the paper.
            i.e., it processes only one out of ``resample_rate`` frames.
            Default: 16.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Default: 8.
        slow_pathway (dict): Configuration of slow branch, should contain
            necessary arguments for building the specific type of pathway
            and:
            type (str): type of backbone the pathway bases on.
            lateral (bool): determine whether to build lateral connection
            for the pathway.Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=True, depth=50, pretrained=None,
                conv1_kernel=(1, 7, 7), dilations=(1, 1, 1, 1),
                conv1_stride_t=1, pool1_stride_t=1, inflate=(0, 0, 1, 1))

        fast_pathway (dict): Configuration of fast branch, similar to
            `slow_pathway`. Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=False, depth=50, pretrained=None, base_channels=8,
                conv1_kernel=(5, 7, 7), conv1_stride_t=1, pool1_stride_t=1)
    """

    def __init__(self,
                 pretrained_encoder=None,
                 pretrained=None,
                 use_resize=True,
                 resample_rate=8,
                 speed_ratio=8,
                 channel_ratio=8,
                 motion_scale = 0.1,
                 slow_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=True,
                     conv1_kernel=(1, 7, 7),
                     dilations=(1, 1, 1, 1),
                     conv1_stride_t=1,
                     pool1_stride_t=1,
                     inflate=(0, 0, 1, 1)),
                 fast_pathway=dict(
                     type='resnet3d',
                     depth=50,
                     pretrained=None,
                     lateral=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7),
                     conv1_stride_t=1,
                     pool1_stride_t=1)):
        super().__init__()
        self.pretrained = pretrained
        self.pretrained_encoder = pretrained_encoder
        self.motion_scale = motion_scale

        self.encoder = ResNet3dSlowFast(pretrained=pretrained_encoder,
                                        dec_use=True,
                                        resample_rate=resample_rate,
                                        speed_ratio=speed_ratio,
                                        channel_ratio=channel_ratio,
                                        slow_pathway=slow_pathway,
                                        fast_pathway=fast_pathway)
        self.slow_decoder = ResNetDec3d(base_channels=64,
                                        depth=50,
                                        pretrained=None,
                                        skip=False, # changed
                                        num_stages=4,
                                        conv1_kernel=(1, 7, 7),
                                        dilations=(1, 1, 1, 1),
                                        conv1_stride_t=1,
                                        pool1_stride_t=1,
                                        inflate=(1, 1, 0, 0))
        num_freedom = 4 if use_resize else 3
        self.fast_regression = RegressionHead(num_freedom=num_freedom,
                                              in_channels=8*32*2, # changed
                                              use_two_layers=True, # changed
                                              use_resize=use_resize,)
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            # Directly load 3D model.
            load_checkpoint(self, self.pretrained, strict=True, logger=logger, prefix='backbone')
        elif self.pretrained is None:
            # print(self.pretrained_encoder)
            # if self.pretrained_encoder is None:
            self.encoder.init_weights()
            # Init two branch seperately.
            self.slow_decoder.init_weights()
            self.fast_regression.init_weights()
            # self.slow_path.init_weights() # to be continued
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x1, x2):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted
                by the backbone.
        """
        x1_slow, x1_fast = self.encoder(x1)
        x2_slow, x2_fast = self.encoder(x2)
        # print(x_slow.size(), x_fast.size())
        x_slow = torch.cat((x1_slow, x2_slow), dim=1)
        # x_slow [N, 64*16*2, 2, H/16=7, W/16=7]
        x_fast = torch.cat((x1_fast, x2_fast), dim=1)
        # print(x_slow.size(), x_fast.size())
        # x_fast [N, 8*32*2, 16, H/32=4, H/32=4]
        p_static = self.slow_decoder(x_slow)
        # print(torch.min(p_static), torch.max(p_static))
        # p_static [N, 3, 1, H=112, W=112]
        # print(p_static.size())
        p_motion = self.fast_regression(x_fast) * self.motion_scale  #/10.0
        # print(p_motion[0])
        # print(p_static.size(), p_motion.size())
        # p_motion [N, 16, num_freedom],
        # print(p_motion.size())
        # exit()
        p = self.code_p(p_static, p_motion)
        return p

    def code_p(self, p_static, p_motion):
        p = []
        p_static = p_static.squeeze(dim=-3)
        # last_p_frame = p_static.squeeze(dim=-3)
        # print(last_p_fr   ame.size()) #[N, C, H, W]
        prefix_dx, prefix_dy, prefix_r_mag, prefix_r_size = 0.0, 0.0, 1.0, 1.0
        # print(p_motion[0, :, 0] / self.motion_scale) # [N, 16, 4]
        for frame_idx in range(16):
            motion = p_motion[:, frame_idx, :]
            dx, dy, r_mag, r_size = motion[:, 0], motion[:, 1], motion[:, 2]+1, motion[:, 3]+1  # range(-1,1)
            prefix_dx = dx + prefix_dx # [N]
            # print(prefix_dx)
            prefix_dy = dy + prefix_dy
            prefix_r_mag = r_mag * prefix_r_mag
            # print(motion[:, 1])  # [N]
            # print(motion[:, 0])
            prefix_r_size = r_size * prefix_r_size
            # current_p_frame = self.code_p_frame(last_p_frame, p_motion[:,frame_idx,:])
            # current_p_frame = self.code_p_frame(p_static, dx, dy, r_mag, r_size)
            current_p_frame = self.code_p_frame(p_static, prefix_dx, prefix_dy, prefix_r_mag, prefix_r_size)
            # current_p_frame = torch.clamp(current_p_frame, -1.0, 1.0)
            p.append(current_p_frame.unsqueeze(dim=-3))
            # last_p_frame = current_p_frame
        p = torch.cat(p, dim=-3)
        return p

    def create_grid_flow(self, flow, N, H, W):
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W, 1).repeat(N, 1, 1, 1)
        yy = yy.view(1, H, W, 1).repeat(N, 1, 1, 1)
        grid = torch.cat((xx, yy), 3).float()
        if flow.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flow
        # print(vgrid[0,:,:,0])
        # print(vgrid[0,:,:,1])
        # scale grid to [-1,1]
        vgrid[:, :, :, 0] = 2.0*vgrid[:, :, :, 0].clone() / max(W-1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0*vgrid[:, :, :, 1].clone() / max(H-1, 1) - 1.0
        # print(vgrid[0, :, :, 0])
        # print(vgrid[0, :, :, 1])
        return vgrid

    def create_grid_resize(self, scale_factor, N, H, W):
        # N, _, H, W = list(x.size())
        vgrid = []
        for batch_idx in range(N):
            xx = torch.arange(-(W - 1) / 2.0, (W - 1) / 2.0 + 1).view(1, -1).repeat(H, 1)
            yy = torch.arange(-(H - 1) / 2.0, (H - 1) / 2.0 + 1).view(-1, 1).repeat(1, W)
            if scale_factor.is_cuda:
                xx = xx.cuda()
                yy = yy.cuda()
            xx = xx * (1.0 / scale_factor[batch_idx]) + (W - 1) / 2.0
            yy = yy * (1.0 / scale_factor[batch_idx]) + (H - 1) / 2.0
            # print(xx)
            # print(yy)
            xx = xx.view(1, H, W, 1)  # .repeat(N, 1, 1, 1)
            yy = yy.view(1, H, W, 1)  # .repeat(N, 1, 1, 1)
            vgrid.append(torch.cat((xx, yy), 3).float())
        vgrid = torch.cat(vgrid, dim=0)
        # print(vgrid.size())

        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0].clone() / max(W - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1].clone() / max(H - 1, 1) - 1.0
        # print(vgrid[0, :, :, 0])
        # print(vgrid[0, :, :, 1])
        return vgrid


    def code_p_frame(self, last_p_frame,  dx, dy, r_mag, r_size):
        # last_p_frame # [N, 3, H=112, W=112]
        # motion # [N, num_freedom]
        N, _,  H, W = list(last_p_frame.size())
        # dx, dy, m = motion[:, 0], motion[:, 1], motion[:, 2]  # range(-1,1)

        dx = dx * W #).to(torch.int) #[N]
        dy = dy * H #).to(torch.int) #[N]
        flow = torch.cat((dx.view(-1, 1), dy.view(-1, 1)), dim=1)
        flow = flow.view(N, 1, 1, 2).repeat(1, H, W, 1) # flow size [N,H,W,2]
        vgrid1 = self.create_grid_flow(flow, N, H, W)
        # print(vgrid1.size())
        vgrid2 = self.create_grid_resize(r_size, N, H, W)
        # print(vgrid2.size())
        r_mag = r_mag.view((N, 1, 1, 1)).expand_as(last_p_frame) # range(0,2)
        #print(flow)
        #print(m)
        p_frame = nn.functional.grid_sample(last_p_frame*r_mag, vgrid1, align_corners=True)
        p_frame = nn.functional.grid_sample(p_frame, vgrid2, align_corners=True)
        mask = torch.ones(p_frame.size())
        if flow.is_cuda:
            mask = mask.cuda()
        # print('mask:', mask[0, 0, :, :])
        mask = nn.functional.grid_sample(mask, vgrid1, align_corners=True)
        mask = nn.functional.grid_sample(mask, vgrid2, align_corners=True)
        # print('mask:', mask[0,0,:,:])
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        p_frame = p_frame * mask
        return p_frame


class DecomposeGDTarget(nn.Module, metaclass=ABCMeta):
    """Class for MP Generator model.
    Args:
        input_flow (bool): whether the inputs are flow or RGB.
    """

    def __init__(self,
                 recognizer,
                 use_resize=True,
                 pretrained_encoder=None,
                 pretrained=None,
                 motion_scale = 0.1,
                 resample_rate=8,
                 speed_ratio=8,
                 channel_ratio=8,
                 p_max=10,
                 class_level=False):
        super().__init__()
        self.p_max = p_max
        self.class_level = class_level
        #self.motion_scale = motion_scale
        if use_resize:
            print('We are using resize parameter :)')
        else:
            print('We are not using resize parameter :(')

        self.generator = DecomposeTargetedG(pretrained_encoder=pretrained_encoder,
                                     pretrained=pretrained,
                                     use_resize=use_resize,
                                     motion_scale=motion_scale,
                                     resample_rate=resample_rate,
                                     speed_ratio=speed_ratio,
                                     channel_ratio=channel_ratio)
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
        output = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values())))
        )
        return output

    def forward(self, data_batch, return_loss=False, output_p=False, **kwargs):
        imgs = data_batch['imgs1']
        label = data_batch['label2']
        # print(data_batch['label1'], data_batch['label2'])
        x1, x2 = data_batch['imgs1'], data_batch['imgs2']
        p = self.generator(x1, x2)
        #print(torch.min(p), torch.max(p))
        # print(p.size())
        if self.class_level:  #
            p1 = p[torch.randperm(p.size()[0])]
            # assert torch.any(p != p1)
            p = p1
        # print(p.size())

        p_imgs = imgs + p * self.p_max
        aux_info = {}
        outputs = self.recognizer(torch.unsqueeze(p_imgs, 1), label, return_loss=return_loss, **aux_info)
        if output_p:
            return outputs, p*self.p_max

        return outputs


def test_motion_code():
    generator = DecomposeG(pretrained_encoder=None,
                           pretrained=None)  # .cuda()
    print('model built')
    # summary(res_dec, input_size=(64*16, 2, 7, 7))
    # x = torch.tensor(np.random.rand(10, 3, 16, 112, 112), dtype=torch.float)#.cuda()
    # p = generator(x)
    # print(p.size())
    # torch.Size([10, 3, 1, 112, 112])
    p_static = torch.tensor(np.random.rand(10, 3, 1, 112, 112), dtype=torch.float)
    p_motion = torch.tensor(np.random.rand(10, 16, 3), dtype=torch.float)
    # print(p_motion.size())
    # print(p_motion[:,1,:].size())
    # p = generator.code_p(p_static, p_motion)
    # print(p.size())
    last_p_frame1 = [
        [11, 12, 13, 14, 15],
        [21, 22, 23, 24, 25],
        [31, 32, 33, 34, 35],
        [41, 42, 43, 44, 45],
        [51, 52, 53, 54, 55]]
    last_p_frame2 = [
        [110, 120, 130, 140, 150],
        [210, 220, 230, 240, 250],
        [310, 320, 330, 340, 350],
        [410, 420, 430, 440, 450],
        [510, 520, 530, 540, 550]]
    last_p_frame1 = torch.tensor(last_p_frame1, dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=0)
    last_p_frame2 = torch.tensor(last_p_frame2, dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=0)
    last_p_batch = torch.cat([last_p_frame1, last_p_frame2], dim=0)
    # print(last_p_batch.size()) # [2, 1, H, w]
    # print(last_p_batch[0, 0, :, :])
    # print(last_p_batch[1, 0, :, :])
    motion = torch.tensor([[0.2, 0.2, 0, 0], [0, 0, -0.5, -0.5]], dtype=torch.float)
    # print(motion.size())  # [2,4]
    dx, dy, r_mag, r_size = motion[:, 0], motion[:, 1], motion[:, 2] + 1, motion[:, 3] + 1
    p_frame = generator.code_p_frame(last_p_batch, dx, dy, r_mag, r_size)
    print(p_frame[0, 0, :, :])
    print(p_frame[1, 0, :, :])

    """
    p_static = torch.unsqueeze(last_p_batch, dim=-3)
    p_motion = torch.unsqueeze(motion, dim=-2).repeat([1, 16, 1])
    p_final = generator.code_p(p_static, p_motion)  # [N, 1, 16, H, W]
    # print(p_final.size())
    print(p_final[0, 0, 0, :, :])
    print(p_final[0, 0, 1, :, :])
    print(p_final[1, 0, 0, :, :])
    print(p_final[1, 0, 1, :, :])
    """

def test_loss_curve(generator, loss_func, test_freedom='dx', device='cpu'):
    # assert test_freedom in {'dx','dy','dm','ds'}
    # a random static frame
    frame_static = torch.randn(size=[5, 5], dtype=torch.float, requires_grad=False, device=device) * 10  # [h=5,w=5]
    frame_static_batch = frame_static.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)  # [1,1,1,5,5] # [n=1,c=1,t=1,h=5,w=5]

    # the initialization of the motion prior
    frame_motion_x = torch.zeros([16, 1], requires_grad='dx' in test_freedom, device=device)
    frame_motion_y = torch.zeros([16, 1], requires_grad='dy' in test_freedom, device=device)
    frame_motion_m = torch.zeros([16, 1], requires_grad='dm' in test_freedom, device=device)
    frame_motion_s = torch.zeros([16, 1], requires_grad='ds' in test_freedom, device=device)

    # the target motion prior & optimizer set-up
    target_motion_x = torch.zeros([16, 1], requires_grad=False)
    target_motion_y = torch.zeros([16, 1], requires_grad=False)
    target_motion_m = torch.zeros([16, 1], requires_grad=False)
    target_motion_s = torch.zeros([16, 1], requires_grad=False)

    trainable_paras = []
    if 'dx' in test_freedom:
        target_motion_x = torch.randn([16, 1], requires_grad=False) * 2 - 1
        trainable_paras.append(frame_motion_x)
    elif 'dy' in test_freedom:
        target_motion_y = torch.randn([16, 1], requires_grad=False) * 2 - 1
        trainable_paras.append(frame_motion_y)
    elif 'dm' in test_freedom:
        target_motion_m = torch.randn([16, 1], requires_grad=False) * 2 - 1
        trainable_paras.append(frame_motion_m)
    elif 'ds' in test_freedom:
        target_motion_s = torch.randn([16, 1], requires_grad=False) * 2 - 1
        trainable_paras.append(frame_motion_s)
    else:
        raise SystemError
    optimizer = torch.optim.SGD(trainable_paras, lr=0.0005)

    # calculate target_p_final
    target_motion = torch.cat([target_motion_x, target_motion_y, target_motion_m, target_motion_s], dim=-1)
    target_motion_batch = target_motion.unsqueeze(dim=0).to(device)
    target_p_final = generator.code_p(frame_static_batch, target_motion_batch).detach()
    target_p_final = target_p_final.squeeze()  # [1,1,16,5,5]

    loss_curve = []
    for t in range(2500):
        # print(frame_motion_x.requires_grad, frame_motion_y.requires_grad,
        #       frame_motion_m.requires_grad, frame_motion_s.requires_grad)

        frame_motion = torch.cat([frame_motion_x, frame_motion_y, frame_motion_m, frame_motion_s], dim=-1)
        # print(frame_motion.requires_grad)
        # exit()

        frame_motion_batch = frame_motion.unsqueeze(dim=0)  # [n=1,T=16,4]
        p_final = generator.code_p(frame_static_batch, frame_motion_batch)
        loss = loss_func(p_final.squeeze(), target_p_final)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())
    return loss_curve

def test_grid_sample():
    N, C, H, W = 10, 3, 112, 112
    last_frame = torch.randn(size=[N,C,H,W])*10
    target_vgrid = torch.randn(size=[N,H,W,2])
    vgrid = torch.zeros(size=[N,H,W,2],requires_grad=True)
    target_frame = nn.functional.grid_sample(last_frame, target_vgrid,
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([vgrid], lr=0.0005)
    loss_curve = []
    for t in range(2500):
        next_frame = nn.functional.grid_sample(last_frame, vgrid,
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
        loss = loss_func(target_frame, next_frame)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())
    return loss_curve



if __name__ == '__main__':
    # test_motion_code()
    # exit()

    # test_cnt = 10
    # for idx in range(test_cnt):
    #     print('Calculating the {:d}-th curve'.format(idx))
    #     loss_curve = test_grid_sample()
    #     plt.plot(loss_curve)
    # #plt.ylim(0, 150)
    # plt.savefig('train_grid_sample.png')
    # plt.clf()
    # exit()

    device = 'cpu'
    torch.set_printoptions(precision=2)
    generator = DecomposeG(pretrained_encoder=None,
                           pretrained=None).to(device)
    loss_func = torch.nn.MSELoss()
    test_cnt = 3
    for idx in range(test_cnt):
        print('Calculating the {:d}-th curve'.format(idx))
        loss_curve = test_loss_curve(generator,loss_func,test_freedom='dx',device=device)
        plt.plot(loss_curve)
    plt.ylim(0,150)
    plt.savefig('train_dx_noise.png')
    plt.clf()
    # for idx in range(test_cnt):
    #     print('Calculating the {:d}-th curve'.format(idx))
    #     loss_curve = test_loss_curve(generator,loss_func,test_freedom='dy',device=device)
    #     plt.plot(loss_curve)
    # plt.ylim(0, 150)
    # plt.savefig('train_dy_roll.png')
    # plt.clf()
    # for idx in range(test_cnt):
    #     print('Calculating the {:d}-th curve'.format(idx))
    #     loss_curve = test_loss_curve(generator,loss_func,test_freedom='dm',device=device)
    #     plt.plot(loss_curve)
    # plt.ylim(0, 150)
    # plt.savefig('train_dm.png')
    # plt.clf()
    # for idx in range(test_cnt):
    #     print('Calculating the {:d}-th curve'.format(idx))
    #     loss_curve = test_loss_curve(generator,loss_func,test_freedom='ds',device=device)
    #     plt.plot(loss_curve)
    # plt.ylim(0, 150)
    # plt.savefig('train_ds.png')
    # plt.clf()







    # torch.set_printoptions(precision=2)
    # generator = DecomposeG(pretrained_encoder=None,
    #                        pretrained=None)  # .cuda()
    # frame_static = [
    #     [11, 12, 13, 14, 15],
    #     [21, 22, 23, 24, 25],
    #     [31, 32, 33, 34, 35],
    #     [41, 42, 43, 44, 45],
    #     [51, 52, 53, 54, 55]] #[5,5]
    # frame_static = torch.tensor(frame_static, dtype=torch.float, requires_grad=False).unsqueeze(dim=0)  # [c=1,h=5,w=5]
    # frame_static_batch = frame_static.unsqueeze(dim=0).unsqueeze(dim=0)  # [1,1,1,5,5] # [n=1,c=1,t=1,h=5,w=5]
    #
    # frame_motion_xy = torch.zeros([16, 2], requires_grad=True)  # [16, 4]
    # frame_motion_ms = torch.zeros([16, 2], requires_grad=False)
    # # frame_motion = torch.cat([frame_motion_xy, frame_motion_ms], dim=-1)
    # # frame_motion = torch.zeros([16, 4], requires_grad=True)
    # # print(frame_motion.requires_grad)
    # # frame_motion_batch = frame_motion.unsqueeze(dim=0)  # [n=1,T=16,4]
    # optimizer = torch.optim.SGD([frame_motion_xy], lr=0.0005)  # frame_motion_xy,
    # # optimizer = torch.optim.SGD([frame_motion], lr=0.0001)
    # # target_motion_batch = torch.zeros([16,4], requires_grad=False).unsqueeze(dim=0) #[n=1, T=16, 4]
    # # target_motion_batch[0, 8, 0], target_motion_batch[0, 8, 1] = 0.2, 0.2
    # # target_motion_batch[0, 6, 0], target_motion_batch[0, 9, 1] = 0.1, 0.9
    # target_motion_xy = torch.zeros([16, 2], requires_grad=False)  # [16, 4]
    # target_motion_ms = torch.zeros([16, 2], requires_grad=False)
    # target_motion = torch.cat([target_motion_xy, target_motion_ms], dim=-1)
    # target_motion_batch = target_motion.unsqueeze(dim=0)
    # # print(target_motion_batch[0])
    # target_p_final = generator.code_p(frame_static_batch, target_motion_batch).detach()
    # target_p_final = target_p_final.squeeze() #[1,1,16,5,5]
    # # for t in range(16):
    # #     print(target_p_final[t,:,:])
    # loss_func = torch.nn.MSELoss()
    #
    #
    # for t in range(2500):
    #     frame_motion = torch.cat([frame_motion_xy, frame_motion_ms], dim=-1)
    #     frame_motion_batch = frame_motion.unsqueeze(dim=0)  # [n=1,T=16,4]
    #     p_final = generator.code_p(frame_static_batch, frame_motion_batch)
    #     #loss = torch.norm(p_final, dim=[-1, -2])
    #     #print(loss.size())  # [1,1,16]
    #     #loss = loss.mean()
    #     loss = loss_func(p_final.squeeze(), target_p_final)
    #     if t%100 == 0:
    #         print('========{:d}=========='.format(t))
    #         print(frame_motion)
    #         #print(p_final[0, 0, 0, :, :])
    #         print(loss)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()




















