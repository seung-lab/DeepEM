from __future__ import print_function

import torch
import torch.nn as nn

import emvision
from emvision.models.layers import BilinearUp
from deepem.models.layers import Conv, Scale


def create_model(opt):
    width = [16,32,64,128,256,512]
    if opt.group > 0:
        # Group normalization
        core = emvision.models.rsunet_gn(width=width[:opt.depth], group=opt.group)
    else:
        # Batch (instance) normalization
        core = emvision.models.RSUNet(width=width[:opt.depth])
    return Model(core, opt.in_spec, opt.out_spec, is_onnx=opt.onnx)


class InputBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InputBlock, self).__init__()
        self.add_module('conv', Conv(in_channels, out_channels, kernel_size))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec, kernel_size, is_onnx=False):
        super(OutputBlock, self).__init__()
        for k, v in out_spec.items():
            out_channels = v[-4]
            if k == 'embedding':
                self.add_module(k, nn.Sequential(
                    Conv(in_channels, out_channels, kernel_size, bias=True),
                    Scale()
                ))
            else:
                self.add_module(k, nn.Sequential(
                    Conv(in_channels, out_channels, kernel_size, bias=True)
                ))
        self.is_onnx = is_onnx

    def forward(self, x):
        outs = {k: m(x) for k, m in self.named_children()}
        # ONNX doesn't support dictionary.
        if self.is_onnx:
            outs = [x[1] for x in sorted(outs.items(), key=lambda x: x[0])]
        return outs


class Model(nn.Sequential):
    """
    Residual Symmetric U-Net with down/upsampling in/output.
    """
    def __init__(self, core, in_spec, out_spec, is_onnx=False):
        super(Model, self).__init__()

        assert len(in_spec)==1, "model takes a single input"
        in_channels = 1
        out_channels = 16
        io_kernel = (1,5,5)

        self.add_module('in', InputBlock(in_channels, out_channels, io_kernel))
        self.add_module('core', core)
        self.add_module('out', OutputBlock(out_channels, out_spec, io_kernel, is_onnx=is_onnx))
