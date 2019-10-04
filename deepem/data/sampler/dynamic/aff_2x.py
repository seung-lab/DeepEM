import numpy as np

from augmentor import Augment
from dataprovider3 import DataProvider, Dataset

from deepem.data.sampler.aff import Sampler as SubSampler


def get_spec(in_spec, out_spec):
    spec = dict()
    # Input spec
    for k, v in in_spec.items():
        spec[k] = v[-3:]
    # Output spec
    for k, v in out_spec.items():
        dim = tuple(v[-3:])
        spec[k] = dim
        spec[k+'_mask'] = dim
    return spec


def f(spec):
    spec = dict()
    for k, v in spec.items():
        spec[k] = tuple(v[0], v[1]//2, v[2]//2)
    return spec


class Sampler(object):
    def __init__(self, data, spec, is_train, aug=None, prob=None):
        sampler1 = SubSampler(data, spec, is_train, aug=aug, prob=prob)
        sampler2 = SubSampler(data, f(spec), is_train, aug=aug, prob=prob)
        self.samplers = [sampler1, sampler2]

    def __call__(self):
        idx = np.random.choice(len(self.datasets), size=1)
        sampler = self.samplers[idx[0]]
        return sampler()
