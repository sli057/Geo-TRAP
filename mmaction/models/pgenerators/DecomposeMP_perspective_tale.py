import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import _load_checkpoint, load_state_dict , load_checkpoint
from mmcv.utils import print_log

from ...utils import get_root_logger
from ..registry import BACKBONES
from mmaction.models import ResNet3d, ResNet3dSlowFast
from .resnet3d_dec import ResNetDec3d
from .regression_head import RegressionHead, AffineRegressionHead, \
    AffineRegressionHeadNoneCu, AffineRegressionHeadNoneCuAdjust, \
    PerspectiveRegressionHead
from abc import ABCMeta

import numpy as np
import matplotlib.pyplot as plt
import kornia


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
                 transform_type = None,
                 pretrained_encoder=None,
                 pretrained=None,

                 resample_rate=8,
                 speed_ratio=8,
                 channel_ratio=8,

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
        #num_freedom = 6
        self.fast_regression = PerspectiveRegressionHead(transform_type=transform_type, in_channels=8*32,
                                                         regularization=regularization)
        #AffineRegressionHeadNoneCu(num_freedom=num_freedom,in_channels=8*32)
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
        #print(p_static)
        # print(torch.min(p_static), torch.max(p_static))
        # p_static [N, 3, 1, H=112, W=112]
        # print(self.regularization)
        if self.regularization:
            p_motion, regularization_loss = self.fast_regression(x_fast) #* self.motion_scale  #/10.0
            #print(regularization_loss)
        else:
            p_motion = self.fast_regression(x_fast)
        #regularization_loss = torch.mean(torch.norm(p_motion, dim=-1))
        # print(p_motion[0]/self.motion_scale, regularization_loss, self.motion_scale)
        # print(p_static.size(), p_motion.size())
        # p_motion [N, 16, num_freedom],
        p = self.code_p(p_static, p_motion)

        if self.regularization:
            return p, regularization_loss * self.regularization_alpha
        else:
            return p



    def code_p(self, p_static, p_motion):
        # print(p_static.size(), p_motion.size()) # [N, 3, 1, H=112, W=112] #[N,16,3,3]
        # return
        p = []
        p_static = p_static.squeeze(dim=-3) #[N,C,H,W]
        N, C, H, W = list(p_static.size())
        N, T, m1, m2 = list(p_motion.size())
        assert m1 == 3 and m2 == 3

        #print(p_motion[0,:,:])
        for frame_idx in range(T):
            motion = p_motion[:, frame_idx, :, :] # [N,3,3]
            #print(motion)
            #motion = motion.view(N, 3, 3)
            current_p_frame = kornia.geometry.transform.warp_perspective(src=p_static, M=motion, dsize=[H,W],
                                                                          mode='bilinear', padding_mode='zeros',
                                                                          align_corners=True)
            p.append(current_p_frame.unsqueeze(dim=-3))
        p = torch.cat(p, dim=-3)
        return p



class DecomposeGDUnTarget(nn.Module, metaclass=ABCMeta):
    """Class for MP Generator model.
    Args:
        input_flow (bool): whether the inputs are flow or RGB.
    """

    def __init__(self,
                 recognizer,
                 transform_type,
                 use_resize=None, #not used
                 pretrained_encoder=None,
                 pretrained=None,
                 regularization=False,
                 regularization_alpha=1,
                 motion_scale=None, #not used
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


        self.generator = DecomposeG(transform_type=transform_type,
                                    pretrained_encoder=pretrained_encoder,
                                    pretrained=pretrained,
                                    #use_resize=use_resize,
                                    regularization=regularization,
                                    regularization_alpha=regularization_alpha,
                                    #motion_scale=motion_scale,
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
        #p = p[:, :, 0, :, :].unsqueeze(dim=-3).repeat([1, 1, 16, 1, 1])
        p_imgs = imgs + p * self.p_max
        aux_info = {}
        outputs = self.recognizer(torch.unsqueeze(p_imgs, 1), label, return_loss=return_loss, **aux_info)
        if output_p:
            return outputs, p*self.p_max
        if self.regularization:
            return outputs, reg_loss

        return outputs


class DecomposeTargetedG:
    pass


class DecomposeGDTarget:
    pass


class DecomposeGDOneTarget:
    pass


def test_warp_perspecive(generator, device='cpu'):
    frame_static = [
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25],
            [31, 32, 33, 34, 35],
            [41, 42, 43, 44, 45],
            [51, 52, 53, 54, 55]]
    frame_static = torch.tensor(frame_static, dtype=torch.float, requires_grad=False, device=device)  # [h=5,w=5]
    frame_static_batch = frame_static.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)  # [1,1,1,5,5] # [n=1,c=1,t=1,h=5,w=5]

    # the initialization of the motion prior
    motion_vector_1 = [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]
    motion_vector_2 = [[1, 0, 2],
                       [0, 1, 2],
                       [0, 0, 1]]
    motion_vector_3 = [[2, 0, 0],
                       [0, 2, 0],
                       [0, 0, 1]]
    motion_vector = torch.tensor([motion_vector_1, motion_vector_2, motion_vector_3], dtype=torch.float,
                                 requires_grad=False, device=device) #[T=3, m1=3, m2=3]
    motion_vector = motion_vector.unsqueeze(dim=0)

    p_final = generator.code_p(frame_static_batch, motion_vector) #[n=1, c= 1, t=3, h=5, w=5]
    N, C, T, H, W = list(p_final.size())
    assert N == 1 and C == 1 and T == 3
    for i in range(T):
        print(p_final[0,0,i,:,:])





if __name__ == '__main__':
    device = 'cpu'
    torch.set_printoptions(precision=2)
    generator = DecomposeG(pretrained_encoder=None,
                           pretrained=None).to(device)
    test_warp_perspecive(generator, device=device)
