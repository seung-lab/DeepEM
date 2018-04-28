from __future__ import print_function

import torch
from torch import nn


class MeanLoss(nn.Module):
    def __init__(self, loss_weight):
        super(MeanLoss, self).__init__()
        assert(len(loss_weight)==2)
        self.loss_weight = loss_weight

    def forward(self, x, objs):
        objs = objs.type(torch.cuda.IntTensor)
        assert(objs.data.type()=='torch.cuda.IntTensor' or
               objs.data.type()=='torch.IntTensor')

        # TODO: Do we need to split objects here?

        ids = self.unique_ids(objs)
        idm = dict([(x,i) for i, x in enumerate(ids)])

        vecs = self.generate_vecs(x, objs, ids)
        means = self.generate_means(vecs)
        weights = self.generate_weights(vecs)

        # Compute loss.
        loss_int = self.compute_loss_int(vecs, means, weights)
        loss_ext = self.compute_loss_ext(means, , weights)
        w_int, w_ext = self.loss_weight
        loss = w_int*loss_int + w_ext*loss_ext

        return loss

    def unique_ids(self, objs):
        ids = np.unique(objs.data.cpu().numpy())
        return ids[ids != 0]

    def generate_vecs(self, x, objs, ids):
        vecs = list()
        for i, obj_id in enumerate(ids):

            vecs.append(vec)
        return vecs

    def generate_weights(self, vecs):
        return [1.0] * len(vecs)

    def generate_means(self, vecs):
        return [torch.mean(v, dim=0) for v in vecs]

    def compute_loss_int(self, vecs, means, weights):
        pass

    def compute_loss_ext(self, vecs, means, weights):
        pass
