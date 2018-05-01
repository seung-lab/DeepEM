from __future__ import print_function
import imp
import os

import torch

from deepem.loss.loss import BCELoss
from deepem.loss.affinity import AffinityLoss
from deepem.train.data import Data
from deepem.train.model import Model


def get_criteria(opt):
    criteria = dict()
    for k in opt.out_spec:
        if k == 'affinity':
            criteria[k] = AffinityLoss(opt.edges,
                size_average=opt.size_average,
                margin=opt.margin
            )
        else:
            criteria[k] = BCELoss(size_average=opt.size_average)
    return criteria


def load_model(opt):
    # Create a model.
    mod = imp.load_source('model', opt.model)
    model = Model(mod.create_model(opt), opt)

    # Load from a checkpoint, if any.
    if opt.chkpt_num > 0:
        model = load_chkpt(model, opt)

    # Multi-GPU training
    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model)

    return model.train().cuda()


def load_chkpt(model, opt):
    print("LOAD CHECKPOINT: {} iters.".format(opt.chkpt_num))
    fname = os.path.join(opt.model_dir, "model{}.chkpt".format(opt.chkpt_num))
    model.load(fname)
    return model


def load_data(opt):
    mod = imp.load_source('data', opt.data)
    data_ids = list(set().union(opt.train_ids, opt.val_ids))
    data = mod.load_data(opt.data_dir, data_ids=data_ids)

    # Train
    train_data = {k: data[k] for k in opt.train_ids}
    train_loader = Data(opt, train_data, is_train=True)

    # Validation
    val_data = {k: data[k] for k in opt.val_ids}
    val_loader = Data(opt, val_data, is_train=False)

    return train_loader, val_loader
