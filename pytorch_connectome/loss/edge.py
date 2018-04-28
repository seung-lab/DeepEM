from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_connectome.utils import torch_utils


class EdgeSampler(object):
    def __init__(self, edges, split_boundary=True):
        self.edges = list(edges)
        self.split_boundary = split_boundary

    def generate_edges(self):
        return list(self.edges)

    def generate_true_aff(self, obj, edge):
        o1, o2 = torch_utils.get_pair2(obj, edge)
        if self.split_boundary:
            ret = (((o1 == o2) + (o1 != 0) + (o2 != 0)) == 3)
        else:
            ret = (o1 == o2)
        return ret.type(obj.type())

    def generate_mask_aff(self, mask, edge):
        m1, m2 = torch_utils.get_pair2(mask, edge)
        return (m1 * m2).type(mask.type())


class EdgeCRF(nn.Module):
    def __init__(self, size_average=False, margin=0):
        super(EdgeCRF, self).__init__()
        self.size_average = size_average
        self.margin = np.clip(margin, 0, 1)

    def forward(self, preds, targets, masks):
        assert len(preds)==len(targets)==len(masks)
        loss = 0
        nmsk = 0
        for pred, target, mask in zip(preds, targets, masks):
            l, n = self.cross_entropy(pred, target, mask)
            loss += l
            nmsk += n
        assert nmsk.item() > 0
        if self.size_average:
            try:
                loss = loss / nmsk
                nmsk = torch.tensor(1, dtype=nmsk.type(),
                                       device=nmsk.device)
            except:
                import pdb; pdb.set_trace()
        return loss, nmsk

    def cross_entropy(self, pred, target, mask):
        mask = self.class_balancing(target, mask)
        if self.margin > 0:
            t = 1 - self.margin
            target[torch.eq(target, 1)] = t
            target[torch.eq(target, 0)] = 1 - t
        bce = F.binary_cross_entropy_with_logits
        ret = dict()
        loss = bce(pred, target, weight=mask, size_average=False)
        nmsk = (mask > 0).type(loss.type()).sum()
        return loss, nmsk

    def class_balancing(self, target, mask):
        dtype = mask.type()
        m_int = mask * torch.eq(target, 1).type(dtype)
        m_ext = mask * torch.eq(target, 0).type(dtype)
        n_int = m_int.sum().item()
        n_ext = m_ext.sum().item()
        if n_int > 0 and n_ext > 0:
            m_int *= n_ext/(n_int + n_ext)
            m_ext *= n_int/(n_int + n_ext)
        return (m_int + m_ext).type(dtype)
