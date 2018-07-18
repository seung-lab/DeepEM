from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn


class MeanLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.001, delta_v=0.0,
                       delta_d=1.5):
        super(MeanLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_v = delta_v  # variance (intra-cluster pull force) hinge
        self.delta_d = delta_d  # distance (inter-cluster push force) hinge

    def forward(self, x, objs):
        objs = objs.type(torch.cuda.IntTensor)
        assert objs.type() == 'torch.cuda.IntTensor'

        ids = self.unique_ids(objs)
        idm = dict([(x,i) for i, x in enumerate(ids)])

        vecs = self.generate_vecs(x, objs, ids)
        means = self.generate_means(vecs)
        weights = self.generate_weights(vecs)

        # Compute loss
        loss_int = self.compute_loss_int(vecs, means, weights)
        loss_ext = self.compute_loss_ext(means, weights)
        loss_nrm = self.compute_loss_nrm(means)
        loss = (self.alpha * loss_int) +
               (self.beta * loss_ext)  +
               (self.gamma * loss_nrm)
        return loss

    def unique_ids(self, objs):
        ids = np.unique(objs.cpu().numpy())
        return ids[ids != 0]

    def generate_vecs(self, embedding, objs, ids):
        vecs = list()
        for i, obj_id in enumerate(ids):
            obj = torch.nonzero(objs == int(obj_id))
            z, y, x = obj[:,-3], obj[:,-2], obj[:,-1]
            vec = embedding[0,:,z,y,x].transpose(0,1)  # C x D
            vecs.append(vec)
        return vecs

    def generate_weights(self, vecs):
        return [1.0] * len(vecs)

    def generate_means(self, vecs):
        return [torch.mean(v, dim=0) for v in vecs]

    def compute_loss_int(self, vecs, means, weights):
        zero = lambda: torch.zeros(1).type(torch.cuda.FloatTensor)
        loss = zero()

        if len(vecs) > 0:
            for v, m, w in zip(vecs, means, weights):
                margin = torch.norm(v - m, p=1, dim=1) - self.delta_v
                loss += w * torch.mean(torch.max(margin, zero())**2)
            loss /= max(1.0, len(vecs))

        return loss

    def compute_loss_ext(self, means, weights):
        zero = lambda: torch.zeros(1).type(torch.cuda.FloatTensor)
        loss = zero()

        C = len(means)
        if C > 1:
            ms = torch.stack(means)
            m1 = ms.unsqueeze(0)  # 1 x C x D
            m2 = ms.unsqueeze(1)  # C x 1 x D

            margin = 2 * self.delta_d - torch.norm(m2 - m1, p=1, dim=2)
            margin = margin[1 - torch.eye(C).type(torch.cuda.ByteTensor)]
            numer = torch.sum(torch.max(margin, zero())**2)
            denom = max(1.0, C * (C - 1.0))
            loss = numer / denom

        return loss

    def compute_loss_nrm(self, means):
        loss = torch.mean(torch.norm(torch.stack(means), p=1, dim=1))
        return loss
