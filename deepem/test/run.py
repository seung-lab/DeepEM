import os

import torch

from deepem.test.forward import Forward
from deepem.test.option import Options
from deepem.test.utils import *


def test(opt):
    # Model
    model = load_model(opt)

    # Forward scan
    forward = Forward(opt)

    if opt.gs_input:
        scanner = make_forward_scanner(opt)
        output, aug_out = forward(model, scanner)
        save_output(output, opt, aug_out=aug_out)
    else:
        for dname in opt.data_names:
            scanner = make_forward_scanner(opt, data_name=dname)
            output, _ = forward(model, scanner)
            save_output(output, opt, data_name=dname)


if __name__ == "__main__":
    # Options
    opt = Options().parse()

    # GPU
    if not opt.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id        
        torch.backends.cudnn.benchmark = not opt.no_autotune

    # Make directories.
    if not os.path.isdir(opt.exp_dir):
        os.makedirs(opt.exp_dir)
    if not os.path.isdir(opt.model_dir):
        os.makedirs(opt.model_dir)
    if not os.path.isdir(opt.fwd_dir):
        os.makedirs(opt.fwd_dir)

    # Run inference.
    print("Running inference: {}".format(opt.exp_name))
    test(opt)
