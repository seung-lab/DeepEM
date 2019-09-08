import numpy as np
import os

import dataprovider3.emio as emio


# New CREMI
data_dir = 'flyem/ground_truth/cremi_b/mip1/padded_x256_y256_z16'
data_info = {
    'cremi_b':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'glia': 'glia.h5',
        'msk': 'msk.h5',
        'dir': '',
        'loc': True,
        'glia_ids': [151,177,705,1414],
    },
}


def load_data(base_dir, data_ids=None, **kwargs):
    if data_ids is None:
        data_ids = data_info.keys()
    data = dict()
    base = os.path.expanduser(base_dir)
    dpath = os.path.join(base, data_dir)
    for data_id in data_ids:
        info = data_info[data_id]
        data[data_id] = load_dataset(dpath, data_id, info, **kwargs)
    return data


def load_dataset(dpath, tag, info, class_keys=[], glia_mask=False, **kwargs):
    assert len(class_keys) > 0
    dset = dict()

    # Image
    fpath = os.path.join(dpath, info['dir'], info['img'])
    print(fpath)
    dset['img']  = emio.imread(fpath).astype(np.float32)
    dset['img'] /= 255.0

    # Segmentation
    if 'aff' in class_keys:
        fpath = os.path.join(dpath, info['dir'], info['seg_d3_b0'])
        print(fpath)
        dset['seg'] = emio.imread(fpath).astype(np.uint32)

    # Glia
    if 'glia' in class_keys:
        fpath = os.path.join(dpath, info['dir'], info['glia'])
        print(fpath)
        dset['glia'] = emio.imread(fpath).astype(np.uint8)

    # Mask
    fpath = os.path.join(dpath, info['dir'], 'msk.h5')
    print(fpath)
    msk = emio.imread(fpath).astype(np.bool)
    if glia_mask:
        assert 'seg' in dset
        gmsk = ~np.isin(dset['seg'], info['glia_ids'])
        msk &= gmsk
    dset['msk'] = msk.astype(np.uint8)

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
