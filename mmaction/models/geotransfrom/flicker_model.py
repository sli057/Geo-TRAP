from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm
from mmaction.models import build_model
from mmcv import Config

class FlickerGeneratorUnTarget(nn.Module):
    """Class for MP Generator model.
        Args:
            input_flow (bool): whether the inputs are flow or RGB.
    """

    def __init__(self,
                 norm_cfg=None,
                 init_std=1,
                 ):
        super().__init__()
        # self.norm_cfg = norm_cfg
        # self.init_std = init_std
        #
        # conv_param = dict(
        #     kernel_size=(3, 3, 3),
        #     padding=(1, 1, 1),
        #     conv_cfg=dict(type='Conv3d'),
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=dict(type='ReLU'))
        #
        # input_channel =  3
        # # [2/3, 16, 112, 112]
        # self.conv1a = ConvModule(input_channel, 64, **conv_param)
        # self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # # [ 64, 16, 56, 56]
        # self.conv2a = ConvModule(64, 128, **conv_param)
        # self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # # [ 128, 16, 28, 28]
        # self.conv3a = ConvModule(128, 256, **conv_param)
        # self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.conv4a = ConvModule(256, 256, **conv_param)
        # self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # # [ 256, 16, 7, 7]
        # self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        # # [ 256, 16, 1, 1]
        # self.fc_cls1 = nn.Linear(256, 16, bias=True)
        # self.fc_cls2 = nn.Linear(16, 3, bias=True)
        #
        # # init_weights
        # self.init_weights()
        # self.perturbation = torch.randn([3, 16]).clone().requires_grad_(True).to(device)
        self.perturbation = torch.nn.Parameter(data=torch.rand([3,16])*2-1, requires_grad=True)
        #torch.nn.Module.register_parameter(self, name='perturbaiton', param=self.perturbation)

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             kaiming_init(m)
    #         elif isinstance(m, nn.Linear):
    #             normal_init(m, std=self.init_std)
    #         elif isinstance(m, _BatchNorm):
    #             constant_init(m, 1)

    def forward(self, x):
        # [N, 3, 16, 112, 112]
        # x = self.pool1(self.conv1a(x))
        # x = self.pool2(self.conv2a(x))
        # x = self.pool3(self.conv3a(x))
        # x = self.pool4(self.conv4a(x))
        # # [N, 256, 16, 7, 7]
        # x = self.avg_pool(x)
        # # [N, 256, 16, 1, 1]
        # x = torch.squeeze(torch.squeeze(x, dim=-1), dim=-1).transpose(dim0=-1, dim1=-2)
        # # [N, 16, 256]
        # x = self.fc_cls1(x)
        # x = torch.relu(x)
        # # [N, 16, 16]
        # x = self.fc_cls2(x)
        # x = torch.tanh(x)
        # # [N, 16, 3]
        # x = x.transpose(dim0=-1, dim1=-2).unsqueeze(dim=-1).unsqueeze(dim=-1)
        # # [N, 3, 16, 1, 1]
        # x = x.repeat([1, 1, 1, 112, 112])
        # print(self.parameters())
        # print(self.perturbation[:,0])
        # print(self.perturbation.grad)

        perturbation = torch.tanh(self.perturbation) #self.perturbation #torch.clamp(self.perturbation, -1.0, 1.0)#
        #print(perturbation[0])
        N, C, T, H, W = list(x.size())
        p = perturbation.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=0)
        p = p.repeat([N, 1, 1, H, W])
        return p #torch.clamp(p, min=-1.0, max=1.0)


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
        self.generator = FlickerGeneratorUnTarget(norm_cfg=norm_cfg, init_std=init_std)
        # pretrained recognizer
        self.recognizer = recognizer
        self.recognizer.eval()
        assert isinstance(recognizer, nn.Module)

    def train_step(self, data_batch, return_loss=True, output_p=False, **kwargs):
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
        if output_p:
            outputs, reg_loss, p = self(data_batch, return_loss, output_p, **kwargs)
        else:
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

        loss =  -loss + reg_loss   # loss + reg_loss  #
        log_vars['loss_cls'] =  -log_vars['loss_cls']   #  log_vars['loss_cls']  #
        log_vars['loss'] =  -log_vars['loss'] + reg_loss.item()  #  log_vars['loss'] + reg_loss.item()  #

        output = dict(
            loss= loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values())))
        )
        if output_p:
            return output, p
        return output

    def val_step(self, data_batch, return_loss=True, output_p=False, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        if output_p:
            outputs, _, p = self(data_batch, return_loss, output_p, **kwargs)
            return outputs, p
        outputs, reg_loss = self(data_batch, return_loss, **kwargs)
        loss, log_vars = self.recognizer._parse_losses(outputs)

        loss = loss + reg_loss  # -loss + reg_loss
        log_vars['loss_cls'] = log_vars['loss_cls']  # -log_vars['loss_cls']
        log_vars['loss'] = log_vars['loss'] + reg_loss.item()  # -log_vars['loss'] + reg_loss.item()

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
        aux_info = {}
        if self.recognizer.cls_head.__class__.__name__ == 'TSMHead':
            p_imgs = torch.transpose(p_imgs, dim0=1, dim1=2)
        else:
            p_imgs = torch.unsqueeze(p_imgs, dim=1)
        self.recognizer.eval()
        outputs = self.recognizer(p_imgs, label, return_loss=return_loss, **aux_info)

        reg_loss = self.flicker_regularizer_loss(p)
        #print(reg_loss)
        if output_p:
            return outputs, reg_loss, p * self.p_max
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
    device = 'cpu'
    torch.set_printoptions(precision=2)
    generator = FlickerGeneratorUnTarget()
    print(list(generator.parameters()))

    video_clip = torch.randn([1, 3, 16, 112, 112])  #[N, 3, 16, 112, 112]
    p = generator(video_clip)
    # for t in range(16):
    #     print(p[0, :, t, 0, 0], p[0, :, t, 1, 1])









