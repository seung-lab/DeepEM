from __future__ import print_function
import imp
import os

import torch
from torch.nn.parallel import data_parallel

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
    model = Model(mod.create_model(opt), get_criteria(opt), opt)

    # Load from a checkpoint, if any.
    if opt.chkpt_num > 0:
        model = load_chkpt(model, opt.model_dir, opt.chkpt_num)

    return model.train().cuda()


def load_chkpt(model, fpath, chkpt_num):
    print("LOAD CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
    model.load(fname)
    return model


def save_chkpt(model, fpath, chkpt_num):
    print("SAVE CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
    model.save(fname)


def load_data(opt):
    mod = imp.load_source('data', opt.data)
    data_ids = list(set().union(opt.train_ids, opt.val_ids))
    data = mod.load_data(opt.data_dir, data_ids=data_ids, **opt.data_params)

    # Train
    train_data = {k: data[k] for k in opt.train_ids}
    train_loader = Data(opt, train_data, is_train=True)

    # Validation
    val_data = {k: data[k] for k in opt.val_ids}
    val_loader = Data(opt, val_data, is_train=False)

    return train_loader, val_loader


def forward(model, sample, opt):
    # Forward pass
    if len(opt.gpu_ids) > 1:
        losses, nmasks, preds = data_parallel(model, sample)
    else:
        losses, nmasks, preds = model(sample)

    # Average over minibatch
    losses = {k: v.mean() for k, v in losses.items()}
    nmasks = {k: v.mean() for k, v in nmasks.items()}

    return losses, nmasks, preds
