import os
import numpy as np

import dataprovider3.emio as emio


# Sparse annotation
data_dir = 'flyem/ground_truth/sparse_annotation'
data_info = {
    'img': 'img.h5',
    'msk': 'msk.h5',
    'seg': 'seg.h5',
    'seg_d3_b0': 'seg_d3_b0.h5',
    'dir': 'mip1/padded_x384_y384_z20',
    'loc': True,
}
data_keys = ['svol0{:0>2d}'.format(i+1) for i in range(44)]


def load_data(base_dir, data_ids=None, **kwargs):
    if data_ids is None:
        data_ids = ['sparse_superset']
    data = dict()
    base_dir = os.path.expanduser(base_dir)
    base_dir = os.path.join(base_dir, data_dir)
    if 'sparse_superset' in data_ids:
        for data_id in data_keys:
            dpath = os.path.join(base_dir, data_id)
            if os.path.exists(dpath):
                data[data_id] = load_dataset(dpath, **kwargs)
    return {'sparse_superset': data}


def load_dataset(dpath, **kwargs):
    dset = dict()

    # Image
    fpath = os.path.join(dpath, data_info['dir'], data_info['img'])
    print(fpath)
    dset['img'] = emio.imread(fpath).astype(np.float32)
    dset['img'] /= 255.0

    # Mask
    fpath = os.path.join(dpath, data_info['dir'], data_info['msk'])
    print(fpath)
    dset['msk'] = emio.imread(fpath).astype(np.uint8)

    # Segmentation
    fpath = os.path.join(dpath, data_info['dir'], data_info['seg_d3_b0'])
    print(fpath)
    dset['seg'] = emio.imread(fpath).astype(np.uint32)

    # Background mask
    idx = dset['seg'] == 1
    dset['msk'][idx] = 0

    # Membrane swirl
    idx = dset['seg'] == 2
    dset['seg'][idx] = 0

    # Large lamellar structure
    idx = dset['seg'] == 3
    dset['msk'][idx] = 0

    # Additoinal info
    dset['loc'] = data_info['loc']

    return dset
