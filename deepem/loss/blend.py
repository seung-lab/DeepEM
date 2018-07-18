from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn

from deepem.loss.edge import EdgeLoss
from deepem.loss.mean import MeanLoss


class BlendLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BlendLoss, self).__init__()
        self.edge = EdgeLoss(**kwargs)
        self.mean = MeanLoss(**kwargs)

    def forward(self, x, label, mask):
        if np.random.rand() > 0.5:
            print("EdgeLoss")
            return self.edge(x, label, mask)
        else:
            print("MeanLoss")
            return self.mean(x, label, mask)
