import numpy as np

from deepem.utils.torch_utils import crop_center


class Modifier(object):
    def __call__(self, sample):
        if np.random.rand() < 0.5:
            return sample
        for k, v in sample.items():
            cropsz = (v.shape[-3], v.shape[-2]//2, v.shape[-1]//2)
            sample[k] = crop_center(v, cropsz).contiguous()
        return sample
