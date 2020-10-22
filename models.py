import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

## CODE ADAPTED FROM: https://github.com/sunwj/CAR
## All credits go to sunwj.

### Blocks
class ResBlock(nn.Module): # From EDSR paper. BatchNorm removed because not useful for SR tasks and it saves memory.

    def __init__(self, num_channels, kernel_size=3, act=nn.ReLU(True), res_scale=0.1):
        super(ResBlock, self).__init__()
        self.scale = res_scale
        conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.body = nn.Sequential(conv1, act, conv2)

    def forward(self, x):
        y = self.body(x).mul(self.scale) # scale is here to avoid numerical unstability.
        y += x
        return y


class PoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act=nn.ReLU(True)):
        super(PoolBlock, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        maxpool = nn.MaxPool2d(2)
        self.body = nn.Sequential(conv, act, maxpool)

    def forward(self, x):
        return self.body(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act=nn.ReLU(True)):
        super(ConvBlock, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.body = nn.Sequential(conv, act)

    def forward(self, x):
        return self.body(x)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        self.config = []
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                kernel_size = 3
                m.append(conv(n_feats, 4 * n_feats, kernel_size, bias=bias, padding=(kernel_size // 2)))
                self.config.append(('conv2d', [4*n_feats, n_feats, kernel_size, kernel_size, 1, (kernel_size // 2)]))
                m.append(nn.PixelShuffle(2))
                self.config.append(('pixelshuffle', [2]))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                    self.config.append(('bn', [n_feats]))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                    self.config.append(('relu', [True]))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
                    self.config.append(('prelu', [n_feats, 0.25]))

        elif scale == 3:
            kernel_size = 3
            m.append(conv(n_feats, 9 * n_feats, kernel_size, bias=bias, padding=(kernel_size // 2)))
            self.config.append(('conv2d', [9*n_feats, n_feats, kernel_size, kernel_size, 1, (kernel_size // 2)]))
            m.append(nn.PixelShuffle(3))
            self.config.append('pixelshuffle', [3])
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                self.config.append(('bn', [n_feats]))
            if act == 'relu':
                m.append(nn.ReLU(True))
                self.config.append(('bn', [n_feats]))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
                self.config.append(('prelu', [n_feats, 0.25]))
        else:
            raise NotImplementedError # ToDo: Implement LeakyRELU?

        super(Upsampler, self).__init__(*m)

class MeanShift(nn.Conv2d): # ToDo: faire des stats sur le dataset pour + accurate mean
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


### Models
## EDSR:
# Code adapted from the pytorch implementation of CAR: https://github.com/sunwj/CAR/
class EDSR(nn.Module):
    def __init__(self, n_resblocks=8, n_feats=64, scale=2):
        super(EDSR, self).__init__()
        self.scale_factor = scale

        kernel_size = 3
        act = nn.LeakyReLU(inplace=True)
        self.config = []

        self.sub_mean = MeanShift(1)
        self.config.append(('sub_mean', [(0.4488, 0.4371, 0.4040), (1.0, 1.0, 1.0)]))


        # define head module
        m_head = [nn.Conv2d(3, n_feats, kernel_size, padding=(kernel_size // 2))]
        self.config.append(('conv2d', [n_feats, 3, kernel_size, kernel_size, 1, (kernel_size // 2)]))

        res_scale = 0.1 # For numerical stability otherwise the values diverge

        m_body = [
            ResBlock(
                n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        for _ in range(n_resblocks):
            if isinstance(act, nn.LeakyReLU):
                self.config.append(('resblock_leakyrelu', [n_feats, n_feats, kernel_size, kernel_size, 1, (kernel_size//2), res_scale, 0.01, True])) # default res_scale factor and default negative slope of the leaky_relu after the conv2d params
            elif isinstance(act, nn.ReLU):
                self.config.append(('resblock_relu', [n_feats, n_feats, kernel_size, kernel_size, 1, (kernel_size//2), res_scale, True]))
            else:
                raise NotImplementedError

        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)))
        self.config.append(('conv2d', [n_feats, n_feats, kernel_size, kernel_size, 1, (kernel_size // 2)]))

        # define tail module
        m_tail = [
            Upsampler(nn.Conv2d, scale, n_feats, act=False),
            nn.Conv2d(n_feats, 3, kernel_size, padding=(kernel_size // 2))
        ]

        for conf in m_tail[0].config:
            self.config.append(conf)
        self.config.append(('conv2d', [3, n_feats, kernel_size, kernel_size, 1, (kernel_size // 2)]))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.add_mean = MeanShift(1, sign=1)
        self.config.append(('add_mean', [(0.4488, 0.4371, 0.4040), (1.0, 1.0, 1.0)]))

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        y = self.body(x)
        y += x
        x = self.tail(y)
        x = self.add_mean(x)
        return x

    def getconfig(self):
        return self.config

