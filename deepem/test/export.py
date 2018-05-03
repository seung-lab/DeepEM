from __future__ import print_function
import io
import numpy as np

import onnx

import torch
import torch.nn as nn
import torch.onnx

from deepem.test.option import Options
from deepem.test.utils import *


def export(opt):
    # Model
    torch_model = load_model(opt).cpu()

    # Input to the model
    shape = (1,) + opt.in_spec['input']
    x = torch.randn(shape)
    print("Input shape = {}".format(x.shape))

    # Export the model
    fpath = os.path.join(opt.model_dir, "model{}.onnx".format(opt.chkpt_num))
    torch_out = torch.onnx._export(torch_model, x, fpath, export_params=True)

    # TODO: Check consistency
    # onnx_model = onnx.load(fpath)
    # prepared_backend = onnx_caffe2.backend.prepare(onnx_model)
    # W = {onnx_model.graph.input[0].name: x.numpy()}
    # caffe2_out = prepared_backend.run(W)[0]
    #
    # # Verify the numerical correctness upto 3 decimal places
    # np.testing.assert_almost_equal(torch_out.cpu().numpy(), caffe2_out, decimal=3)
    # print("Exported model has been executed on Caffe2 backend, and the result looks good!")


if __name__ == "__main__":
    # Options
    opt = Options().parse()

    # GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    # Make directories.
    if not os.path.isdir(opt.exp_dir):
        os.makedirs(opt.exp_dir)
    if not os.path.isdir(opt.model_dir):
        os.makedirs(opt.model_dir)
    if not os.path.isdir(opt.fwd_dir):
        os.makedirs(opt.fwd_dir)

    # cuDNN auto-tuning
    torch.backends.cudnn.benchmark = opt.autotune

    # Run inference.
    print("Exporting: {}".format(opt.exp_name))
    export(opt)
