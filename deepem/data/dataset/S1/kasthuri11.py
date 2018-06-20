from __future__ import print_function
import numpy as np
import os

import dataprovider3.emio as emio


# Kasthuri11 dataset
data_info = {
    'AC3':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'AC3/mip0/padded_x512_y512_z32',
        'loc': True,
    },
    'AC4':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'AC4/mip0/padded_x512_y512_z32',
        'loc': True,
    },
}


def load_data(data_dir, data_ids=None, **kwargs):
    if data_ids is None:
        data_ids = data_info.keys()
    data = dict()
    dpath = os.path.expanduser(data_dir)
    for data_id in data_ids:
        info = data_info[data_id]
        data[data_id] = load_dataset(dpath, data_id, info)
    return data


def load_dataset(dpath, tag, info):
    dset = dict()

    # Image
    fpath = os.path.join(dpath, info['dir'], info['img'])
    print(fpath)
    img = emio.imread(fpath).astype('float32') / 255.0
    dset['img'] = img

    # Segmentation
    fpath = os.path.join(dpath, info['dir'], info['seg'])
    print(fpath)
    seg = emio.imread(fpath).astype('uint32')
    dset['seg'] = seg

    # Train mask
    if tag == 'AC4':
        fpath = os.path.join(dpath, info['dir'], 'msk_train.h5')
        print(fpath)
        dset['msk_train'] = emio.imread(fpath).astype('uint8')
        fpath = os.path.join(dpath, info['dir'], 'msk_val.h5')
        print(fpath)
        dset['msk_val'] = emio.imread(fpath).astype('uint8')
    else:
        fpath = os.path.join(dpath, info['dir'], info['msk'])
        print(fpath)
        dset['msk'] = emio.imread(fpath).astype('uint8')

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
