from __future__ import print_function
import imp
import os

import torch
from torch.nn.parallel import data_parallel

import deepem.loss as loss
from deepem.train.data import Data
from deepem.train.model import Model


def get_criteria(opt):
    criteria = dict()
    for k in opt.out_spec:
        if k == 'affinity' or k == 'long_range':
            if k == 'affinity':
                edges = [(0,0,1),(0,1,0),(1,0,0)]
            else:
                edges = list(opt.edges)
            assert len(edges) > 0
            params = dict(opt.loss_params)
            params['size_average'] = False
            criteria[k] = loss.AffinityLoss(edges,
                criterion=getattr(loss, opt.loss)(**params),
                size_average=opt.size_average,
                class_balancing=opt.class_balancing
            )
        else:
            params = dict(opt.loss_params)
            # params['margin0'] = 0
            # params['margin1'] = 0
            # params['inverse'] = False
            criteria[k] = getattr(loss, 'BCELoss')(**params)
    return criteria


def load_model(opt):
    # Create a model.
    mod = imp.load_source('model', opt.model)
    model = Model(mod.create_model(opt), get_criteria(opt), opt)

    if opt.pretrain:
        model.load(opt.pretrain)
    if opt.chkpt_num > 0:
        model = load_chkpt(model, opt.model_dir, opt.chkpt_num)

    return model.train().cuda()


def load_optimizer(opt, trainable):
    # Create an optimizer.
    optimizer = getattr(torch.optim, opt.optim)(trainable, **opt.optim_params)

    if not opt.pretrain and opt.chkpt_num > 0:
        n = opt.chkpt_num
        fname = os.path.join(opt.model_dir, "model{}.chkpt".format(n))
        chkpt = torch.load(fname)
        if 'optimizer' in chkpt:
            print("LOAD OPTIM STATE: {} iters.".format(n))
            optimizer.load_state_dict(chkpt['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    print(optimizer)
    return optimizer


def load_chkpt(model, fpath, chkpt_num):
    print("LOAD CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
    model.load(fname)
    return model


def save_chkpt(model, fpath, chkpt_num, optimizer):
    print("SAVE CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
    state = {'iter': chkpt_num,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, fname)


def load_data(opt):
    mod = imp.load_source('data', opt.data)
    data_ids = list(set().union(opt.train_ids, opt.val_ids))
    data = mod.load_data(opt.data_dir, data_ids=data_ids, **opt.data_params)

    # Train
    train_data = {k: data[k] for k in opt.train_ids}
    if opt.train_prob:
        prob = dict(zip(opt.train_ids, opt.train_prob))
    else:
        prob = None
    train_loader = Data(opt, train_data, is_train=True, prob=prob)

    # Validation
    val_data = {k: data[k] for k in opt.val_ids}
    if opt.val_prob:
        prob = dict(zip(opt.val_ids, opt.val_prob))
    else:
        prob = None
    val_loader = Data(opt, val_data, is_train=False, prob=prob)

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
