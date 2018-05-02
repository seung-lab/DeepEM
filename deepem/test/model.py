from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    """
    Model wrapper for inference.
    """
    def __init__(self, model, opt):
        super(Model, self).__init__()
        self.model = model
        self.in_spec = dict(opt.in_spec)
        self.pretrain = opt.pretrain

    def forward(self, sample):
        inputs = [sample[k] for k in sorted(self.in_spec)]
        preds = self.model(*inputs)
        return {k: F.sigmoid(x) for k, x in preds.items()}

    def load(self, fpath):
        state_dict = torch.load(fpath)
        if self.pretrain:
            model_dict = self.model.state_dict()
            state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(state_dict)
