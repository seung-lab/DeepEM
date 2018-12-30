from __future__ import print_function
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class BCELoss(nn.Module):
    """
    Binary cross entropy loss with logits.
    """
    def __init__(self, size_average=True, margin0=0, margin1=0, inverse=True,
                       **kwargs):
        super(BCELoss, self).__init__()
        self.bce = F.binary_cross_entropy_with_logits
        self.size_average = size_average
        self.margin0 = float(np.clip(margin0, 0, 1))
        self.margin1 = float(np.clip(margin1, 0, 1))
        self.inverse = inverse

    def forward(self, input, target, mask):
        # Number of valid voxels
        nmsk = (mask > 0).type(mask.dtype).sum()
        assert nmsk.item() >= 0
        if nmsk.item() == 0:
            loss = 0
            return loss, nmsk

        # Margin
        m0, m1 = self.margin0, self.margin1
        if m0 > 0 or m1 > 0:
            if self.inverse:
                target[torch.eq(target, 1)] = 1 - m1
                target[torch.eq(target, 0)] = m0
            else:
                activ = F.sigmoid(input)
                m_int = torch.ge(activ, 1 - m1) * torch.eq(target, 1)
                m_ext = torch.le(activ, m0) * torch.eq(target, 0)
                mask *= 1 - (m_int + m_ext).type(mask.dtype)

        loss = self.bce(input, target, weight=mask, size_average=False)

        if self.size_average:
            loss = loss / nmsk.item()
            nmsk = torch.tensor([1], dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk


class MSELoss(nn.Module):
    """
    Mean squared error loss with (or without) logits.
    """
    def __init__(self, size_average=True, margin0=0, margin1=0, logits=True,
                       **kwargs):
        super(MSELoss, self).__init__()
        self.mse = F.mse_loss
        self.size_average = size_average
        self.margin0 = float(np.clip(margin0, 0, 1))
        self.margin1 = float(np.clip(margin1, 0, 1))
        self.logits = logits

    def forward(self, input, target, mask):
        # Number of valid voxels
        nmsk = (mask > 0).type(mask.type()).sum()
        assert nmsk.item() >= 0
        if nmsk.item() == 0:
            loss = 0
            return loss, nmsk

        activ = F.sigmoid(input) if self.logits else input

        # Margin
        m0, m1 = self.margin0, self.margin1
        if m0 > 0 or m1 > 0:
            m_int = torch.ge(activ, 1 - m1) * torch.eq(target, 1)
            m_ext = torch.le(activ, m0) * torch.eq(target, 0)
            mask *= 1 - (m_int + m_ext).type(mask.dtype)

        loss = self.mse(activ, target, reduce=False)
        loss = (loss * mask).sum()

        if self.size_average:
            loss = loss / nmsk.item()
            nmsk = torch.tensor([1], dtype=nmsk.dtype, device=nmsk.device)

        return loss, nmsk
