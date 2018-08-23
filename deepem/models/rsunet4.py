from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn

import emvision
from emvision.models.layers import BilinearUp
from deepem.models.layers import Conv


def create_model(opt):
    width = [16,32,64,128,256,512]
    if opt.width:
        opt.depth = len(opt.width)
        width = opt.width
    core = emvision.models.rsunet_v3.RSUNet(width=width[:opt.depth])
    return Model(core, opt.in_spec, opt.out_spec)


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

        spec = OrderedDict(sorted(out_spec.items(), key=lambda x: x[0]))
        outs = []
        for k, v in spec.items():
            out_channels = v[-4]
            outs.append(nn.Sequential(
                Conv(in_channels, out_channels, kernel_size, bias=True),
                BilinearUp(out_channels, out_channels)
            ))
        self.outs = nn.ModuleList(outs)
        self.keys = spec.keys()

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        return {k: out(x) for k, out in zip(self.keys, self.outs)}


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling in/output.
    """
    def __init__(self, core, in_spec, out_spec):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = 1
        out_channels = 16
        io_kernel = (1,5,5)

        self.add_module('in', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel))
