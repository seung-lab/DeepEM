from __future__ import print_function

from augmentor import Augment
from dataprovider3 import DataProvider

from pytorch_connectome.data.sampler.utils import recompute_CC


def get_data_ids(is_train):
    if is_train:
        return ['stitched_vol19-vol34',
                'stitched_vol40-vol41',
                'vol101',
                'vol102',
                'vol103',
                'vol104',
                'vol401',
                'vol502',
                'vol503']
    else:
        return ['stitched_vol19-vol34']


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
    def __init__(self, data, spec, is_train, aug=None):
        data_ids = get_data_ids(is_train)
        self.is_train = is_train
        self.build(data, data_ids, spec, aug)

    def __call__(self):
        sample = self.dataprovider()
        return self.postprocess(sample)

    def postprocess(self, sample):
        assert('affinity' in sample)
        sample['affinity'] = recompute_CC(sample['affinity'])
        sample = Augment.to_tensor(sample)
        return self.to_float32(sample)

    def to_float32(self, sample):
        for k, v in sample.items():
            sample[k] = v.astype('float32')
        return sample

    def build(self, data, data_ids, spec, aug):
        dp = DataProvider(spec)
        for key in data_ids:
            dp.add_dataset(self.build_dataset(key, data[key]))
        dp.set_augment(aug)
        dp.set_imgs(['input'])
        self.dataprovider = dp

    def build_dataset(self, key, data):
        img = data['img']
        seg = data['seg']
        msk = self.get_mask(data)
        loc = data['loc']
        # Create Dataset.
        dset = Dataset()
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
