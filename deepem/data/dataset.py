from __future__ import print_function
import numpy as np

import torch
import torch.utils.data


def worker_init_fn(worker_id):
    # Each worker already has its own random state (Torch).
    seed = torch.IntTensor(1).random_()[0]
    # print("worker ID = {}, seed = {}".format(worker_id, seed))
    np.random.seed(seed)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sampler, size):
        super(Dataset, self).__init__()
        self.sampler = sampler
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.sampler()
