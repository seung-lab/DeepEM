from __future__ import print_function

from augmentor import Augment
from dataprovider3 import DataProvider


def get_spec(in_spec, out_spec):
    spec = dict()
    for k, v in in_spec.items():
        spec[k] = v[-3:]
    for k, v in out_spec.items():
        spec[k] = v[-3:]
        spec[k+'_mask'] = v[-3:]
    return spec


class Sampler(object):
    def __init__(self, data, spec, aug=None):
        self.build(data, spec, aug)

    def __call__(self):
        sample = self.dataprovider()
        return self.postprocess(sample)

    def postprocess(self, sample):
        return Augment.to_tensor(sample)

    def build(self, data, spec, aug):
        dp = DataProvider(spec)
        for k, v in data.items():
            dp.add_dataset(self.build_dataset(k, v))
        dp.set_augment(aug)
        dp.set_imgs(['input'])
        self.dataprovider = dp

    def build_dataset(self, key, data):
        dset = Dataset()
        dset.add_data(key='input', data=data['img'])
        dset.add_data(key='object', data=data['seg'])
        dset.add_mask(key='object_mask', data=data['msk'], loc=data['loc'])
        return dset
