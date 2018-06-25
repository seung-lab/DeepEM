from __future__ import print_function
import imp
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import DataLoader


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


class Data(object):
    def __init__(self, opt, data, is_train=True, prob=None):
        self.build(opt, data, is_train, prob)

    def __call__(self):
        sample = next(self.dataiter)
        for k in sample:
            is_input = k in self.inputs
            sample[k].requires_grad_(is_input)
            sample[k] = sample[k].cuda(non_blocking=(not is_input))
        return sample

    def requires_grad(self, key):
        return self.is_train and (key in self.inputs)

    def build(self, opt, data, is_train, prob):
        # Data augmentation
        if opt.augment:
            mod = imp.load_source('augment', opt.augment)
            aug = mod.get_augmentation(is_train, **opt.aug_params)
        else:
            aug = None

        # Data sampler
        mod = imp.load_source('sampler', opt.sampler)
        spec = mod.get_spec(opt.in_spec, opt.out_spec)
        sampler = mod.Sampler(data, spec, is_train, aug, prob=prob)

        # Data loader
        size = (opt.max_iter - opt.chkpt_num) * opt.batch_size
        dataset = Dataset(sampler, size)
        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size,
                                num_workers=opt.num_workers,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)

        # Attributes
        self.dataiter = iter(dataloader)
        self.inputs = opt.in_spec.keys()
        self.is_train = is_train
