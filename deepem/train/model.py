from __future__ import print_function

import torch
import torch.nn as nn

from deepem.loss.loss import BCELoss
from deepem.loss.affinity import AffinityLoss
from deepem.train.utils import get_criteria


class Model(nn.Module):
    """
    Model wrapper for training.
    """
    def __init__(self, model, opt):
        super(Model, self).__init__()
        self.model = model
        self.in_spec = opt.in_spec
        self.out_spec = opt.out_spec
        self.criteria = get_criteria(opt)
        self.pretrain = opt.pretrain

    def forward(self, sample):
        # Forward pass
        inputs = [sample[k] for k in sorted(self.in_spec)]
        preds = self.model(*inputs)
        # Loss evaluation
        losses, nmasks = self.eval_loss(preds, sample)
        return losses, nmasks, preds

    def eval_loss(self, preds, sample):
        loss, nmsk = dict(), dict()
        for k in self.out_spec:
            target = sample[k]
            mask = sample[k + '_mask']
            loss[k], nmsk[k] = self.criteria(preds[k], target, mask)
        return loss, nmsk

    def save(self, fpath):
        torch.save(self.model.state_dict(), fpath)

    def load(self, fpath):
        state_dict = torch.load(fpath)
        if self.pretrain:
            model_dict = self.model.state_dict()
            state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(state_dict)
