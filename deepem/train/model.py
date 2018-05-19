from __future__ import print_function

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model wrapper for training.
    """
    def __init__(self, model, criteria, opt):
        super(Model, self).__init__()
        self.model = model
        self.criteria = criteria
        self.in_spec = dict(opt.in_spec)
        self.out_spec = dict(opt.out_spec)
        self.pretrain = opt.pretrain

    def forward(self, sample):
        # Forward pass
        inputs = [sample[k] for k in sorted(self.in_spec)]
        preds = self.model(*inputs)
        # Loss evaluation
        losses, nmasks = self.eval_loss(preds, sample)
        return losses, nmasks, preds

    def eval_loss(self, preds, sample):
        losses, nmasks = dict(), dict()
        for k in self.out_spec:
            target = sample[k]
            mask = sample[k + '_mask']
            loss, nmsk = self.criteria[k](preds[k], target, mask)
            # TODO: PyTorch 0.4.0 bug
            losses[k] = loss.unsqueeze(0)
            nmasks[k] = nmsk.unsqueeze(0)
        return losses, nmasks

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
