from __future__ import print_function

import torch
import torch.nn as nn

from deepem.loss.edge import EdgeCRF
from deepem.utils import torch_utils


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


class AffinityLoss(nn.Module):
    def __init__(self, edges, size_average=False, margin=0):
        super(AffinityLoss, self).__init__()
        self.sampler = EdgeSampler(edges)
        self.decoder = AffinityLoss.Decoder(edges)
        self.criterion = EdgeCRF(
            size_average=size_average,
            margin=margin
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
            assert(len(edges) > 0)
            self.edges = list(edges)

        def forward(self, x, i):
            num_channels = x.size(-4)
            assert(num_channels == len(self.edges))
            assert(i < num_channels and i >= 0)
            edge = self.edges[i]
            return torch_utils.get_pair_first2(x[...,[i],:,:,:], edge)
