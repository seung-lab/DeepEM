from __future__ import print_function

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
    if opt.group > 0:
        # Group normalization
        core = emvision.models.rsunet_gn(width=width[:opt.depth], group=opt.group)
    else:
        # Batch (instance) normalization
        core = emvision.models.RSUNet(width=width[:opt.depth])
    return Model(core, opt.in_spec, opt.out_spec)


class Input(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Input, self).__init__()
        self.down2x = nn.AvgPool3d((1,2,2))
        self.down4x = nn.AvgPool3d((1,4,4))
        self.conv = Conv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x2 = self.conv(self.down2x(x))
        x4 = self.conv(self.down4x(x))
        return [x2, x4]


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size):
        super(OutputBlock, self).__init__()
        for k, v in out_spec.items():
            self.add_module(k, Conv(in_channels, v[-4], kernel_size, bias=True))

    def forward(self, x):
        return {k: m(x) for k, m in self.named_children()}


class Upsample(nn.Module):
    def __init__(self, up, out_spec):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=up, mode='trilinear')
        # for k, v in out_spec.items():
        #     channels = v[-4]
        #     self.add_module(k, BilinearUp(channels, channels, factor=up))
        self.keys = out_spec.keys()

    def forward(self, x):
        return {k: self.up(x[k]) for k in self.keys}
        # return {k: m(x[k]) for k, m in self.named_children()}


class Output(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size):
        super(Output, self).__init__()
        self.out = OutputBlock(in_channels, out_spec, kernel_size)
        self.up2x = Upsample((1,2,2), out_spec)
        self.up4x = Upsample((1,4,4), out_spec)
        self.keys = out_spec.keys()

    def forward(self, x):
        x2 = self.up2x(self.out(x[0]))
        x4 = self.up4x(self.out(x[1]))
        return {k: (x2[k] + x4[k]) / 2.0 for k in self.keys}


class Model(nn.Module):
    """
    Residual Symmetric U-Net with down/upsampling in/output.
    """
    def __init__(self, core, in_spec, out_spec):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = 1
        out_channels = 16
        io_kernel = (1,5,5)

        self.input = Input(in_channels, out_channels, io_kernel)
        self.core = core
        self.output = Output(out_channels, out_spec, io_kernel)

    def forward(self, x):
        xs = self.input(x)
        ys = [self.core(x) for x in xs]
        return self.output(ys)
