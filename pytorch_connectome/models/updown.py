from __future__ import print_function
from collections import OrderedDict
import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import emvision
from emvision.models.layers import BilinearUp
from emvision.models.utils import pad_size


def create_model(opt):
    raise NotImplementedError


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                       bias=False):
        super(Conv, self).__init__()
        padding = pad_size(kernel_size, 'same')
        self.conv = nn.Conv3d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal(self.conv.weight)
        if bias:
            nn.init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('down', nn.AvgPool3d((1,2,2)))
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size):
        super(OutputBlock, self).__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Sort outputs by name.
        spec = OrderedDict(sorted(out_spec.items(), key=lambda x: x[0]))
        outs = []
        for k, v in spec.items():
            out_channels = v[-4]
            outs.append(nn.Sequential(
                Conv(in_channels, out_channels, kernel_size, bias=True),
                BilinearUp(out_channels, out_channels)
            ))
        self.outs = nn.ModuleList(outs)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        return [out(x) for out in self.outs]


class RSUNet(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling in/output.
    """
    def __init__(self, in_spec, out_spec, depth, **kwargs):
        super(RSUNet, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = list(in_spec.values())[0][0]

        width = [16,32,64,128,256,512]

        self.add_module('in', InputBlock(in_channels, 16, (1,5,5)))
        self.add_module('core', emvision.models.RSUNet(width=width[:depth]))
        self.add_module('out', OutputBlock(16, out_spec, (1,5,5)))
