from __future__ import print_function

import torch
import torch.nn as nn

import emvision
from emvision.models.utils import pad_size

from deepem.utils import torch_utils


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                       bias=False):
        super(Conv, self).__init__()
        padding = pad_size(kernel_size, 'same')
        self.conv = nn.Conv3d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class Crop(nn.Module):
    def __init__(self, cropsz):
        self.cropsz = tuple(cropsz)

    def forward(self. x):
        if self.cropsz is not None:
            for k, v in x.items():
                x[k] = torch_utils.crop_center(v, self.cropsz)
        return x
