from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn

from .regression_head import RegressionHead
from mmaction.models import ResNet3d, ResNet3dSlowFast
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm
from mmaction.models import build_model
from mmcv import Config
import numpy as np
from torchsummary import summary

class FlickerGeneratorUnTarget(nn.Module):
        """Class for MP Generator model.
            Args:
                input_flow (bool): whether the inputs are flow or RGB.
        """

        def __init__(self,
                     pretrained_encoder='checkpoints/slowfast_r50_video_4x16x1_256e_kinetics400_rgb_20200826-f85b90c5.pth',
                     resample_rate=1,
                     speed_ratio=1,
                     channel_ratio=1,
                     slow_pathway=dict(
                         type='resnet3d',
                         depth=50,
                         pretrained=None,
                         lateral=False,
                         conv1_kernel=(1, 7, 7),
                         dilations=(1, 1, 1, 1),
                         conv1_stride_t=1,
                         pool1_stride_t=1,
                         inflate=(0, 0, 1, 1)),
                     fast_pathway = dict(
                        type='resnet3d',
                        depth=50,
                        pretrained=None,
                        lateral=False,
                        base_channels=8,
                        conv1_kernel=(5, 7, 7),
                        conv1_stride_t=1,
                        pool1_stride_t=1)):
            super().__init__()
            self.pretrained_encoder = pretrained_encoder

            self.encoder = ResNet3dSlowFast(pretrained=pretrained_encoder,
                                            dec_use=True,
                                            resample_rate=resample_rate,
                                            speed_ratio=speed_ratio,
                                            channel_ratio=channel_ratio,
                                            slow_pathway=slow_pathway,
                                            fast_pathway=fast_pathway)
            self.decoder = RegressionHead(in_channels=1024, num_freedom=3) #256

        def init_weights(self):
            self.encoder.init_weights()
            self.decoder.init_weights()

        def forward(self, x):
            # x1 = x1.reshape((-1,) + x1.shape[2:])
            # x2 = x2.reshape((-1,) + x2.shape[2:])
            _, _, _ , H, W = list(x.size())
            x_slow, x_fast = self.encoder(x)
            # x_slow [N, 64*16, 2, H/16=7, W/16=7]
            # x_fast [N, 8*32, 16, H/32=4, H/32=4]
            x = x_slow
            p = self.decoder(x).transpose(dim0=-1, dim1=-2) #[N, 3, 16]
            p = p.unsqueeze(dim=-1).unsqueeze(dim=-1)
            p = p.repeat([1, 1, 1, H, W]) #[N, 3, 16, H, W]
            return p

class FlickerGDUnTarget(nn.Module, metaclass=ABCMeta):
    """Class for MP Generator model.
    Args:
        input_flow (bool): whether the inputs are flow or RGB.
    """

    def __init__(self,
                 recognizer,
                 beta_0 = 0.5, #1.0,
                 beta_1 = 0.5,
                 beta_2 = 0.5,
                 norm_cfg=None,
                 init_std=0.1,
                 p_max=10,
                 ):
        super().__init__()
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.norm_cfg = norm_cfg
        self.init_std = init_std
        self.p_max = p_max
        self.generator = FlickerGeneratorUnTarget()
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
        outputs, reg_loss = self(data_batch, return_loss, **kwargs)
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
        loss = -loss + reg_loss
        log_vars['loss_cls'] = -log_vars['loss_cls']
        log_vars['loss'] = -log_vars['loss'] + reg_loss.item()

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
        outputs, reg_loss = self(data_batch, return_loss, **kwargs)
        loss, log_vars = self.recognizer._parse_losses(outputs)

        loss = -loss + reg_loss
        log_vars['loss_cls'] = -log_vars['loss_cls']
        log_vars['loss'] = -log_vars['loss'] + reg_loss.item()

        output = dict(
            loss= loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values())))
        )
        return output

    def forward(self, data_batch, return_loss=False, output_p=False, **kwargs):
        x = data_batch['imgs1']
        # label1, label2 = data_batch['label1'], data_batch['label2']
        imgs = data_batch['imgs1']
        label = data_batch['label1']
        p = self.generator(x)
        # print(p.size())
        p_imgs = imgs + p * self.p_max
        #print(p_imgs.size())
        aux_info = {}
        if self.recognizer.cls_head.__class__.__name__ == 'TSMHead':
            p_imgs = torch.transpose(p_imgs, dim0=1, dim1=2)
        else:
            p_imgs = torch.unsqueeze(p_imgs, dim=1)
        self.recognizer.eval()
        outputs = self.recognizer(p_imgs, label, return_loss=return_loss, **aux_info)
        if output_p:
            return outputs, p * self.p_max
        reg_loss = self.flicker_regularizer_loss(p)
        #print(reg_loss)
        return outputs, reg_loss

    def flicker_regularizer_loss(self, p):
        # print(p[0, 0, 0, 0, 0])
        # [N, 3, 16, H, W]
        norm_reg = torch.mean(p**2) + 1e-12
        p_roll_right = torch.roll(p, 1, dims=-3)
        p_roll_left = torch.roll(p, -1, dims=-3)
        diff_norm_reg = torch.mean((p-p_roll_right)**2) + 1e-12
        laplacian_norm_reg = torch.mean((-2*p+p_roll_right+p_roll_left)**2) + 1e-12
        regularizer_loss = self.beta_1 * norm_reg + self.beta_2 * diff_norm_reg + self.beta_2 * laplacian_norm_reg
        return self.beta_0 * regularizer_loss


if __name__ == '__main__':
    x1 = torch.tensor(np.random.rand(10, 3, 16, 112, 112), dtype=torch.float)
    generator = FlickerGeneratorUnTarget()
    p = generator(x1)
    print(p.size())
    exit()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    summary(generator, (3, 16, 112, 112))
    #total parameter : 11M
    exit()

    device = 'cpu'
    torch.set_printoptions(precision=2)
    generator = FlickerGeneratorUnTarget()
    print(list(generator.parameters()))

    video_clip = torch.randn([1, 3, 16, 112, 112])  #[N, 3, 16, 112, 112]
    p = generator(video_clip)
    # for t in range(16):
    #     print(p[0, :, t, 0, 0], p[0, :, t, 1, 1])









