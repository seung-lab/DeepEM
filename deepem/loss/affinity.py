from __future__ import print_function

import torch
import torch.nn as nn

from deepem.loss.edge import EdgeSampler, EdgeCRF
from deepem.utils import torch_utils


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
