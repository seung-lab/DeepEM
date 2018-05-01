from __future__ import print_function
import imp

import torch
from torch.utils.data import DataLoader

from pytorch_connectome.data.dataset import Dataset, worker_init_fn


class Data(object):
    def __init__(self, opt, data, is_train, device=None):
        self.build(opt, data, is_train, device)

    def __call__(self):
        sample = next(self.dataiter)
        for k in sample:
            sample[k].requires_grad_(self.requires_grad(k))
            sample[k] = sample[k].to(self.device)
            # TODO: Non-blocking data transfer?
        return sample

    def requires_grad(self, key):
        return self.is_train and (key in self.inputs)

    def build(self, opt, data, is_train, device):
        # Data augmentation
        mod = imp.load_source('augment', opt.augment)
        aug = mod.get_augmentation(is_train)

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
        self.device = torch.device('cuda') if device is None else device
