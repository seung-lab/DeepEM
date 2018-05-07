from __future__ import print_function
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class BCELoss(nn.Module):
    """
    Binary cross entropy loss with logits.
    """
    def __init__(self, size_average=True, margin=0):
        super(BCELoss, self).__init__()
        self.criterion = F.binary_cross_entropy_with_logits
        self.size_average = size_average
        self.margin = np.clip(margin, 0, 1)

    def forward(self, input, target, mask):
        # Inverse margin
        if self.margin > 0:
            t = 1 - self.margin
            target[torch.eq(target, 1)] = t
            target[torch.eq(target, 0)] = 1 - t

        loss = self.criterion(input, target, weight=mask, size_average=False)
        nmsk = (mask > 0).type(loss.type()).sum()
        assert(nmsk.item() > 0)

        if self.size_average:
            loss = loss / nmsk
            nmsk = torch.tensor([1], dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk


class MSELoss(nn.Module):
    """
    Mean squared error loss with logits.
    """
    def __init__(self, size_average=True):
        super(MSELoss, self).__init__()
        self.criterion = F.mse_loss
        self.size_average = size_average

    def forward(self, input, target, mask):
        loss = self.criterion(F.sigmoid(input), target, reduce=False)
        loss = (loss * mask).sum()
        nmsk = (mask > 0).type(loss.type()).sum()
        assert(nmsk.item() > 0)

        if self.size_average:
            loss = loss / nmsk
            nmsk = torch.tensor([1], dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk
