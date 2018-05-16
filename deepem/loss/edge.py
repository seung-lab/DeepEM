from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepem.utils import torch_utils


class EdgeSampler(object):
    def __init__(self, max_edge, edges=[]):
        self.max_edge = tuple(max_edge)
        self.edges = list(edges)

    def generate_edges(self, n=32):
        edges = list(self.edges)
        for _ in range(n):
            x = np.random.randint(0, self.max_edge[-1])
            y = np.random.randint(0, self.max_edge[-2])
            z = np.random.randint(0, self.max_edge[-3])
            edge = (z,y,x)
            edge = tuple(int(i * np.random.choice([1,-1])) for i in edge)
            edges.append(edge)
        return edges

    def generate_target(self, objs, mask, edge):
        mask *= (objs != 0).type(mask.dtype)
        true_aff = self.generate_true_aff(objs, edge)
        mask_aff = self.generate_mask_aff(mask, edge)
        return true_aff, mask_aff

    def generate_true_aff(self, objs, edge):
        o1, o2 = torch_utils.get_pair(objs, edge)
        return (o1 == o2).type(objs.dtype)

    def generate_mask_aff(self, mask, edge):
        m1, m2 = torch_utils.get_pair(mask, edge)
        return (m1 * m2).type(mask.dtype)


class EdgeCRF(nn.Module):
    def __init__(self, size_average=False):
        super(EdgeCRF, self).__init__()
        self.size_average = size_average

    def forward(self, preds, targets, masks):
        assert(len(preds)==len(targets)==len(masks))
        loss = 0
        nmsk = 0
        for pred, target, mask in zip(preds, targets, masks):
            l, n = self.cross_entropy(pred, target, mask)
            loss += l
            nmsk += n
        assert(nmsk.item() > 0)
        if self.size_average:
            try:
                loss = loss / nmsk.item()
                nmsk = torch.tensor([1], dtype=nmsk.dtype, device=nmsk.device)
            except:
                # import pdb; pdb.set_trace()
                raise
        return loss, nmsk

    def cross_entropy(self, pred, target, mask):
        # mask = self.class_balancing(target, mask)
        bce = F.binary_cross_entropy_with_logits
        loss = bce(pred, target, weight=mask, size_average=False)
        nmsk = (mask > 0).type(loss.type()).sum()
        return loss, nmsk

    # def class_balancing(self, target, mask):
    #     dtype = mask.type()
    #     m_int = mask * torch.eq(target, 1).type(dtype)
    #     m_ext = mask * torch.eq(target, 0).type(dtype)
    #     n_int = m_int.sum().item()
    #     n_ext = m_ext.sum().item()
    #     if n_int > 0 and n_ext > 0:
    #         m_int *= n_ext/(n_int + n_ext)
    #         m_ext *= n_int/(n_int + n_ext)
    #     return (m_int + m_ext).type(dtype)


class EdgeLoss(nn.Module):
    def __init__(self, max_edge, n_edge=32, edges=[], size_average=False):
        super(EdgeLoss, self).__init__()
        self.sampler = EdgeSampler(max_edge, edges=edges)
        self.n_edge = max(n_edge, 0)
        self.decoder = EdgeLoss.Decoder()
        self.criterion = EdgeCRF(size_average=size_average)

    def forward(self, vec, label, mask):
        pred_affs = list()
        true_affs = list()
        mask_affs = list()
        edges = self.sampler.generate_edges(n=self.n_edge)
        for edge in edges:
            try:
                pred_affs.append(self.decoder(vec, edge))
                t, m = self.sampler.generate_target(label, mask, edge)
                true_affs.append(t)
                mask_affs.append(m)
            except:
                raise
        return self.criterion(pred_affs, true_affs, mask_affs)

    class Decoder(nn.Module):
        def __init__(self):
            super(EdgeLoss.Decoder, self).__init__()

        def forward(self, vec, edge):
            v1, v2 = torch_utils.get_pair(vec, edge)
            return torch_utils.affinity(v1, v2)
