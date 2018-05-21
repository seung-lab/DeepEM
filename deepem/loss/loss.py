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
        self.bce = F.binary_cross_entropy_with_logits
        self.size_average = size_average
        self.margin = np.clip(margin, 0, 1)
        self.inverse = inverse

    def forward(self, input, target, mask):
        # Number of valid voxels
        nmsk = (mask > 0).type(mask.dtype).sum()
        assert(nmsk.item() > 0)

        # Margin
        if self.margin > 0:
            m = self.margin
            if self.inverse:
                target[torch.eq(target, 1)] = 1 - m
                target[torch.eq(target, 0)] = m
            else:
                activ = F.sigmoid(input)
                m_int = torch.ge(activ, 1 - m) * torch.eq(target, 1)
                m_ext = torch.le(activ, m) * torch.eq(target, 0)
                mask *= 1 - (m_int + m_ext).type(mask.dtype)

        loss = self.bce(input, target, weight=mask, size_average=False)

        if self.size_average:
            loss = loss / nmsk.item()
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
        # Number of valid voxels
        nmsk = (mask > 0).type(mask.type()).sum()
        assert(nmsk.item() > 0)

        activ = F.sigmoid(input)

        # Margin
        if self.margin > 0:
            m = self.margin
            m_int = torch.ge(activ, 1 - m) * torch.eq(target, 1)
            m_ext = torch.le(activ, m) * torch.eq(target, 0)
            mask *= 1 - (m_int + m_ext).type(mask.dtype)

        loss = self.mse(activ, target, reduce=False)
        loss = (loss * mask).sum()

        if self.size_average:
            loss = loss / nmsk.item()
            nmsk = torch.tensor([1], dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk


class BCELoss2(nn.Module):
    """
    Binary cross entropy loss with logits.
    """
    def __init__(self, size_average=True, margin=0, inverse=True, **kwargs):
        super(BCELoss2, self).__init__()
        self.bce = F.binary_cross_entropy_with_logits
        self.size_average = size_average
        self.margin = np.clip(margin, 0, 1)
        self.inverse = inverse

    def forward(self, input, target, mask):
        # Margin
        if self.margin > 0:
            m = self.margin
            if self.inverse:
                target[torch.eq(target, 1)] = 1 - m
                target[torch.eq(target, 0)] = m
            else:
                activ = F.sigmoid(input)
                m_int = torch.ge(activ, 1 - m) * torch.eq(target, 1)
                m_ext = torch.le(activ, m) * torch.eq(target, 0)
                mask *= 1 - (m_int + m_ext).type(mask.dtype)

        loss = self.bce(input, target, weight=mask, size_average=self.size_average)
        nmsk = torch.tensor([1], dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk
