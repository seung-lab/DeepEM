from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn


class BoundaryLoss(nn.Module):
    def __init__(self, criterion, class_balancing=False):
        super(BoundaryLoss, self).__init__()
        self.criterion = criterion
        self.balancing = class_balancing

    def forward(self, pred, label, mask):
        target = (label != 0).type(pred.type())
        mask = self.class_balancing(target, mask)
        return self.criterion(pred, target, mask)

    def class_balancing(self, target, mask):
        if not self.balancing:
            return mask
        dtype = mask.type()
        m_int = mask * torch.eq(target, 1).type(dtype)
        m_ext = mask * torch.eq(target, 0).type(dtype)
        n_int = m_int.sum().item()
        n_ext = m_ext.sum().item()
        if n_int > 0 and n_ext > 0:
            m_int *= n_ext/(n_int + n_ext)
            m_ext *= n_int/(n_int + n_ext)
        return (m_int + m_ext).type(dtype)
