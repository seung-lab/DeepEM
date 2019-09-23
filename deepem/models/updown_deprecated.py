import torch
import torch.nn as nn

import emvision
from emvision.models.layers import BilinearUp
from deepem.models.layers import Conv


def create_model(opt):
    if opt.width:
        width = opt.width
        depth = len(width)
    else:
        width = [16,32,64,128,256,512]
        depth = opt.depth
    if opt.group > 0:
        # Group normalization
        core = emvision.models.rsunet_gn(width=width[:depth], group=opt.group)
    else:
        # Batch (instance) normalization
        core = emvision.models.RSUNet(width=width[:depth])
    return Model(core, opt.in_spec, opt.out_spec, width[0])


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('down', nn.AvgPool3d((1,2,2)))
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size):
        super(OutputBlock, self).__init__()
        for k, v in out_spec.items():
            out_channels = v[-4]
            self.add_module(k, nn.Sequential(
                Conv(in_channels, out_channels, kernel_size, bias=True),
                BilinearUp(out_channels, out_channels)
            ))

    def forward(self, x):
        return {k: m(x) for k, m in self.named_children()}


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling in/output.
    """
    def __init__(self, core, in_spec, out_spec, out_channels):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = 1
        io_kernel = (1,5,5)

        self.add_module('in', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel))
