from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn

from deepem.loss.edge import EdgeLoss
from deepem.loss.mean import MeanLoss


class BlendLoss(nn.Module):
    def __init__(self, blending_prop=0.5, **kwargs):
        super(BlendLoss, self).__init__()
        self.prop = float(np.clip(blending_prop, 0, 1))
        self.edge = EdgeLoss(**kwargs)
        self.mean = MeanLoss(**kwargs)

    def forward(self, x, label, mask):
        if np.random.rand() < self.prop:
            return self.edge(x, label, mask)
        else:
            return self.mean(x, label, mask)
