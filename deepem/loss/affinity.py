from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn

from deepem.utils import torch_utils


class EdgeSampler(object):
    def __init__(self, edges):
        self.edges = list(edges)

    def generate_edges(self):
        return list(self.edges)

    def generate_true_aff(self, obj, edge):
        o1, o2 = torch_utils.get_pair(obj, edge)
        ret = (((o1 == o2) + (o1 != 0) + (o2 != 0)) == 3)
        return ret.type(obj.type())

    def generate_mask_aff(self, mask, edge):
        m1, m2 = torch_utils.get_pair(mask, edge)
        return (m1 * m2).type(mask.type())


class EdgeCRF(nn.Module):
    def __init__(self, criterion, size_average=False, class_balancing=False):
        super(EdgeCRF, self).__init__()
        self.criterion = criterion
        self.size_average = size_average
        self.balancing = class_balancing

    def forward(self, preds, targets, masks):
        assert len(preds) == len(targets) == len(masks)
        loss, nmsk = 0, 0
        for pred, target, mask in zip(preds, targets, masks):
            mask = self.class_balancing(target, mask)
            l, n = self.criterion(pred, target, mask)
            loss += l
            nmsk += n
        assert nmsk.item() >= 0

        if nmsk.item() == 0:
            loss = torch.tensor([0]).type(torch.cuda.FloatTensor)
            return loss, nmsk

        if self.size_average:
            assert nmsk.item() > 0
            try:
                loss = loss / nmsk.item()
                nmsk = torch.tensor([1], dtype=nmsk.dtype, device=nmsk.device)
            except:
                import pdb; pdb.set_trace()
                raise

        return loss, nmsk

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


class AffinityLoss(nn.Module):
    def __init__(self, edges, criterion, size_average=False,
                 class_balancing=False):
        super(AffinityLoss, self).__init__()
        self.sampler = EdgeSampler(edges)
        self.decoder = AffinityLoss.Decoder(edges)
        self.criterion = EdgeCRF(
            criterion,
            size_average=size_average,
            class_balancing=class_balancing
        )

    def forward(self, preds, label, mask):
        pred_affs = list()
        true_affs = list()
        mask_affs = list()
        edges = self.sampler.generate_edges()
        for i, edge in enumerate(edges):
            try:
                pred_affs.append(self.decoder(preds, i))
                true_affs.append(self.sampler.generate_true_aff(label, edge))
                mask_affs.append(self.sampler.generate_mask_aff(mask, edge))
            except:
                raise
        return self.criterion(pred_affs, true_affs, mask_affs)

    class Decoder(nn.Module):
        def __init__(self, edges):
            super(AffinityLoss.Decoder, self).__init__()
            assert len(edges) > 0
            self.edges = list(edges)

        def forward(self, x, i):
            num_channels = x.size(-4)
            assert num_channels == len(self.edges)
            assert i < num_channels and i >= 0
            edge = self.edges[i]
            return torch_utils.get_pair_first(x[...,[i],:,:,:], edge)
