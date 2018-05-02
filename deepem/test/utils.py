from __future__ import print_function
import imp
import os

from deepem.test.model import Model


def load_model(opt):
    # Create a model.
    mod = imp.load_source('model', opt.model)
    model = Model(mod.create_model(opt), opt)

    # Load from a checkpoint, if any.
    if opt.chkpt_num > 0:
        model = load_chkpt(model, opt.model_dir, opt.chkpt_num)

    return model.train().cuda()


def load_chkpt(model, fpath, chkpt_num):
    print("LOAD CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
    model.load(fname)
    return model
