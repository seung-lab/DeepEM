from __future__ import print_function
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class BCELoss(nn.Module):
    """
    Binary cross entropy loss with logits.
    """
    def __init__(self, size_average=True, margin=0, inverse=True, **kwargs):
        super(BCELoss, self).__init__()
        self.criterion = F.binary_cross_entropy_with_logits
        self.size_average = size_average
        self.margin = np.clip(margin, 0, 1)
        self.inverse = inverse

    def forward(self, input, target, mask):
        # Margin
        mask2 = None
        if self.margin > 0:
            m = self.margin
            if self.inverse:
                target[torch.eq(target, 1)] = 1 - m
                target[torch.eq(target, 0)] = m
            else:
                m_int = torch.ge(input, 1 - m) * torch.eq(target, 1)
                m_ext = torch.le(input, m) * torch.eq(target, 0)
                mask2 = 1 - (m_int + m_ext).type(mask.dtype)

        loss = self.criterion(input, target, weight=mask, size_average=False)
        loss = loss * mask2 if mask2 else loss
        nmsk = (mask > 0).type(loss.dtype).sum()
        assert(nmsk.item() > 0)

        if self.size_average:
            loss = loss / nmsk
            nmsk = torch.tensor([1], dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk


class MSELoss(nn.Module):
    """
    Mean squared error loss with logits.
    """
    def __init__(self, size_average=True, margin=0, **kwargs):
        super(MSELoss, self).__init__()
        self.mse = F.mse_loss
        self.size_average = size_average

    def forward(self, input, target, mask):
        input = F.sigmoid(input)

        # Margin
        mask2 = None
        if self.margin > 0:
            m_int = torch.ge(input, 1 - self.margin) * torch.eq(target, 1)
            m_ext = torch.le(input, self.margin) * torch.eq(target, 0)
            mask2 = 1 - (m_int + m_ext).type(mask.dtype)

        loss = self.mse(input, target, reduce=False)
        loss = (loss * mask * mask2).sum() if mask2 else (loss * mask).sum()
        nmsk = (mask > 0).type(loss.type()).sum()
        assert(nmsk.item() > 0)

        if self.size_average:
            loss = loss / nmsk
            nmsk = torch.tensor([1], dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk
