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
        if k == 'affinity':
            dim = tuple(d+2 for d in dim)
        spec[k] = dim
        spec[k+'_mask'] = dim
    return spec


class Sampler(object):
    def __init__(self, data, spec, is_train, aug=None, prob=None,
                 recompute=True):
        self.is_train = is_train
        self.recompute = recompute
        self.build(data, spec, aug, prob)

    def __call__(self):
        sample = self.dataprovider()
        return self.postprocess(sample)

    def postprocess(self, sample):
        assert('affinity' in sample)
        if self.recompute:
            sample['affinity'] = recompute_CC(sample['affinity'])
        sample = Augment.to_tensor(sample)
        return self.to_float32(sample)

    def to_float32(self, sample):
        for k, v in sample.items():
            sample[k] = v.astype('float32')
        return sample

    def build(self, data, spec, aug, prob):
        dp = DataProvider(spec)
        keys = data.keys()
        for k in keys:
            dp.add_dataset(self.build_dataset(k, data[k]))
        dp.set_augment(aug)
        dp.set_imgs(['input'])
        if prob:
            dp.set_sampling_weights(p=[prob[k] for k in keys])
        else:
            dp.set_sampling_weights(p=None)
        self.dataprovider = dp
        print(dp)

    def build_dataset(self, tag, data):
        img = data['img']
        seg = data['seg']
        loc = data['loc']
        msk = self.get_mask(data)
        # Create Dataset.
        dset = Dataset(tag=tag)
        dset.add_data(key='input', data=img)
        dset.add_data(key='affinity', data=seg)
        dset.add_mask(key='affinity_mask', data=msk, loc=loc)
        return dset

    def get_mask(self, data):
        key = 'msk_train' if self.is_train else 'msk_val'
        if key in data:
            return data[key]
        assert('msk' in data)
        return data['msk']
