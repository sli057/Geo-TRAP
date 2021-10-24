import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import numpy as np


class PerspectiveRegressionHead(nn.Module):
    """The regression head for motion vector generation.

    Args:
        transform_type (str): different types of geo transformation.
        in_channels (int): Number of channels in input feature.
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 transform_type,
                 in_channels,
                 illu=False,
                 regularization=True,
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01):

        super().__init__()
        self.transform_type = transform_type
        self.regularization = regularization
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.illu = illu


        if self.transform_type == 'translation':
            num_freedom = 2
        elif self.transform_type == 'dilation':
            num_freedom = 1
        elif self.transform_type == 'rigid':
            num_freedom = 3
        elif self.transform_type == 'translation_dilation':
            num_freedom = 3
        elif self.transform_type == 'similarity':
            num_freedom = 4
        elif self.transform_type == 'affine':
            num_freedom = 6
        elif self.transform_type == 'projective':
            num_freedom = 8
        elif self.transform_type == 'no_transformation':
            num_freedom = 8 # Not useful
        else:
            print('no transformation type:', self.transform_type)
            exit()
        print('Geo-transformation type: ', self.transform_type)

        if self.illu:
            num_freedom += 1
            print('using illumination change')

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            print('You may want to use pooling')
            exit()

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.num_freedom = num_freedom
        self.fc_cls1 = nn.Linear(16*in_channels, 16*in_channels//8, bias=True) # 32*8 --> 32 --> 6
        self.fc_cls2 = nn.Linear(in_channels//8, num_freedom, bias=True)
        self.a11, self.a22, self.a33 = 1, 1, 1
        self.a12, self.a13, self.a21, self.a23, self.a31, self.a32 = 0, 0, 0, 0, 0, 0

    def identity_transformation_weights(self):
        if self.transform_type == 'translation':
            weights = [self.a13, self.a23]
        elif self.transform_type == 'dilation':
            weights = [1]
        elif self.transform_type == 'rigid':
            weights = [0, self.a13, self.a23]
        elif self.transform_type == 'translation_dilation':
            weights = [1, self.a13, self.a23]
        elif self.transform_type == 'similarity':
            weights = [0, 1, self.a13, self.a23]
        elif self.transform_type == 'affine':
            weights = [self.a11, self.a12, self.a13, self.a21, self.a22, self.a23]
        elif self.transform_type == 'projective':
            weights = [self.a11, self.a12, self.a13, self.a21, self.a22, self.a23, self.a31, self.a32]
        elif self.transform_type == 'no_transformation':
            weights= [self.a11, self.a12, self.a13, self.a21, self.a22, self.a23, self.a31, self.a32]


        return weights



    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls1, std=self.init_std)
        bias = self.identity_transformation_weights()
        if self.illu:
            bias += [0]
        #bias += np.random.normal(loc=0, scale=1, size=len(bias)) # key!!!! scale 0.01 not working.
        identity_init(self.fc_cls2, std=self.init_std, bias=bias)
        #normal_init(self.fc_cls2)#, std=self.init_std, bias=torch.tensor(bias))

    def vector2perspectiveM(self, x):
        N, T, d = list(x.size())
        a11_batch = torch.ones(size=(N, T), dtype=torch.float, device=x.device) * self.a11
        a12_batch = torch.ones(size=(N, T), dtype=torch.float, device=x.device) * self.a12
        a13_batch = torch.ones(size=(N, T), dtype=torch.float, device=x.device) * self.a13
        a21_batch = torch.ones(size=(N, T), dtype=torch.float, device=x.device) * self.a21
        a22_batch = torch.ones(size=(N, T), dtype=torch.float, device=x.device) * self.a22
        a23_batch = torch.ones(size=(N, T), dtype=torch.float, device=x.device) * self.a23
        a31_batch = torch.ones(size=(N, T), dtype=torch.float, device=x.device) * self.a31
        a32_batch = torch.ones(size=(N, T), dtype=torch.float, device=x.device) * self.a32
        a33_batch = torch.ones(size=(N, T), dtype=torch.float, device=x.device) * self.a33

        # transform to perspective transformation matrix
        if self.transform_type == 'translation':
            a13_batch, a23_batch = x[:, :, 0], x[:, :, 1] # x[n,t=16,d=2]
        elif self.transform_type == 'dilation':
            a11_batch = x[:, :, 0]
            a22_batch = x[:, :, 0]
        elif self.transform_type == 'rigid':
            a11_batch = torch.cos(x[:, :, 0])
            a12_batch = - torch.sin(x[:, :, 0])
            a21_batch = torch.sin(x[:, :, 0])
            a22_batch = torch.cos(x[:, :, 0])
            a13_batch, a23_batch = x[:, :, 1], x[:, :, 2]
        elif self.transform_type == 'translation_dilation':
            a11_batch = x[:, :, 0]
            a22_batch = x[:, :, 0]
            a13_batch, a23_batch = x[:, :, 1], x[:, :, 2]
        elif self.transform_type == 'similarity':
            a11_batch = torch.cos(x[:, :, 0]) * x[:, :, 1]
            a12_batch = - torch.sin(x[:, :, 0]) * x[:, :, 1]
            a21_batch = torch.sin(x[:, :, 0]) * x[:, :, 1]
            a22_batch = torch.cos(x[:, :, 0]) * x[:, :, 1]
            a13_batch, a23_batch = x[:, :, 2], x[:, :, 3]
        elif self.transform_type == 'affine':
            a11_batch = x[:, :, 0]
            a12_batch = x[:, :, 1]
            a13_batch = x[:, :, 2]
            a21_batch = x[:, :, 3]
            a22_batch = x[:, :, 4]
            a23_batch = x[:, :, 5]
        elif self.transform_type == 'projective':
            a11_batch = x[:, :, 0]
            a12_batch = x[:, :, 1]
            a13_batch = x[:, :, 2]
            a21_batch = x[:, :, 3]
            a22_batch = x[:, :, 4]
            a23_batch = x[:, :, 5]
            a31_batch = x[:, :, 6]
            a32_batch = x[:, :, 7]
        else: # self.transform_type == 'projective':
            pass
        row1 = torch.stack([a11_batch, a12_batch, a13_batch], dim=-1) #[N,T,3]
        row2 = torch.stack([a21_batch, a22_batch, a23_batch], dim=-1)
        row3 = torch.stack([a31_batch, a32_batch, a33_batch], dim=-1)
        x_out = torch.stack([row1, row2, row3], dim=-2)

        return x_out

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, 8*32, 16, H/32=3, H/32=3]
        x = self.avg_pool(x)
        # [N, 8*32, 16, 1, 1]
        x = torch.squeeze(torch.squeeze(x, dim=-1), dim=-1).transpose(dim0=-1, dim1=-2)
        # [N, 16, 8*32]
        if self.dropout is not None: # False:  #
            x = self.dropout(x)
        N, T, C = list(x.size())
        x = x.contiguous().view(N, -1) # [N, 16*8*32]
        x = self.fc_cls1(x)
        x = torch.relu(x)
        x = x.view(N, T, -1)
        x = self.fc_cls2(x) # [N,T,d]
        if self.illu:
            illu = x[:,:,-1]
            x = x[:,:,:-1]
        x_out = self.vector2perspectiveM(x)
        if self.regularization:
            identity_transform_weights = self.identity_transformation_weights()
            x_identity = torch.tensor(identity_transform_weights, device=x.device, dtype=x.dtype)
            regularization_loss = torch.mean(torch.norm(x, dim=-1))  #-x_identity, dim=-1))  #
            if self.illu:
                #print(regularization_loss,torch.mean(torch.norm(torch.abs(illu)-1) ))
                regularization_loss += 0.1*torch.mean(torch.norm(torch.abs(illu)-1))
                # print(x_out.size(), illu.size(), regularization_loss.size())
                # exit()
                # torch.Size([15, 16, 3, 3]) torch.Size([15, 16]) torch.Size([])
                return x_out, illu, regularization_loss
            return x_out, regularization_loss
        if self.illu:
            return x_out, illu
        return x_out

def identity_init(module, mean=0, std=1, bias=[1, 0, 0, 0, 1, 0]):
    # if False:  # np.sum(bias) == 0: #Not sure
    #     if len(bias) == 2:
    #         std = 1
    #     nn.init.normal_(module.weight, mean, std)
    # else:
    #     module.weight.data.zero_()
    nn.init.normal_(module.weight, mean, std)
    # if len(bias) == 8:
    #     bias += np.random.normal(loc=0, scale=0.01, size=len(bias)) #?
    #     nn.init.normal_(module.weight, mean, std)
    module.bias.data.copy_(torch.tensor(bias, dtype=torch.float))


if __name__ == '__main__':
    N, T = 2, 3
    head = PerspectiveRegressionHead(transform_type='projective', in_channels=10)
    d = 8
    x = torch.rand(size=[N, T, d], requires_grad=True)
    x_out = head.vector2perspecitveM(x)
    print(x.size(), x_out.size())
    for i in range(N):
        for j in range(T):
            print('===========')
            print(x[i, j, :])
            print(x_out[i, j, :])




