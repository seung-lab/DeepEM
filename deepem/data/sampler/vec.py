from __future__ import print_function

from augmentor import Augment
from dataprovider3 import DataProvider, Dataset

from deepem.data.sampler.utils import recompute_CC


def get_spec(in_spec, out_spec):
    spec = dict()
    for k, v in in_spec.items():
        spec[k] = v[-3:]
    for k, v in out_spec.items():
        dim = tuple(v[-3:])
        spec[k] = dim
        spec[k+'_mask'] = dim
    return spec


class Sampler(object):
    def __init__(self, data, spec, is_train, aug=None):
        self.is_train = is_train
        self.build(data, spec, aug)

    def __call__(self):
        sample = self.dataprovider()
        return self.postprocess(sample)

    def postprocess(self, sample):
        assert('embedding' in sample)
        sample['embedding'] = recompute_CC(sample['embedding'])
        sample = Augment.to_tensor(sample)
        return self.to_float32(sample)

    def to_float32(self, sample):
        for k, v in sample.items():
            sample[k] = v.astype('float32')
        return sample

    def build(self, data, spec, aug):
        dp = DataProvider(spec)
        for k, v in data.items():
            dp.add_dataset(self.build_dataset(k, v))
        dp.set_augment(aug)
        dp.set_imgs(['input'])
        self.dataprovider = dp

    def build_dataset(self, key, data):
        img = data['img']
        seg = data['seg']
        loc = data['loc']
        msk = self.get_mask(data)
        # Create Dataset.
        dset = Dataset()
        dset.add_data(key='input', data=img)
        dset.add_data(key='embedding', data=seg)
        dset.add_mask(key='embedding_mask', data=msk, loc=loc)
        return dset

    def get_mask(self, data):
        key = 'msk_train' if self.is_train else 'msk_val'
        if key in data:
            return data[key]
        assert('msk' in data)
        return data['msk']
