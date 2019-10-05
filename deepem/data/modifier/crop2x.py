import numpy as np

from deepem.utils import torch_utils


class Modifier(object):
    def __call__(self, sample):
        if np.random.rand() < 0.5:
            return sample
        for k, v in sample:
            cropsz = (v.shape[-3], v.shape[-2]//2, v.shape[-1]//2)
            sample[k] = torch_utils.crop_center(v, cropsz)
        return sample
