from __future__ import print_function
import imp

import torch
from torch.utils.data import DataLoader

from deepem.data.dataset import Dataset, worker_init_fn


class Data(object):
    def __init__(self, opt, data, is_train):
        self.build(opt, data, is_train)

    def __call__(self):
        sample = next(self.dataiter)
        for k in sample:
            is_input = k in self.inputs
            sample[k].requires_grad_(is_input)
            sample[k] = sample[k].cuda(non_blocking=(not is_input))
        return sample

    def requires_grad(self, key):
        return self.is_train and (key in self.inputs)

    def build(self, opt, data, is_train):
        # Data augmentation
        mod = imp.load_source('augment', opt.augment)
        aug = mod.get_augmentation(is_train, **opt.aug_params)

        # Data sampler
        mod = imp.load_source('sampler', opt.sampler)
        spec = mod.get_spec(opt.in_spec, opt.out_spec)
        sampler = mod.Sampler(data, spec, is_train, aug)

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
