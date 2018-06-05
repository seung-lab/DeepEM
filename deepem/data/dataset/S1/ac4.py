from __future__ import print_function
import numpy as np
import os

import dataprovider3.emio as emio


# AC4 dataset
ac4_info = {
    'train':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': None,
        'dir': '',
        'loc': True,
    },
}


def load_data(data_dir, data_ids=None, pad_size=(0,0,0), **kwargs):
    if data_ids is None:
        data_ids = ac4_info.keys()
    data = dict()
    dpath = os.path.expanduser(data_dir)
    for data_id in data_ids:
        info = ac4_info[data_id]
        data[data_id] = load_dataset(dpath, data_id, info, pad_size)
    return data


def load_dataset(dpath, tag, info, pad_size):
    dset = dict()

    pad = any(x > 0 for x in pad_size)
    pad_width = [(x//2,x//2) for x in pad_size]

    # Image
    fpath = os.path.join(dpath, info['dir'], info['img'])
    print(fpath)
    img = emio.imread(fpath).astype('float32') / 255.0
    if pad:
        img = np.pad(img, pad_width, 'reflect')
    dset['img'] = img

    # Segmentation
    fpath = os.path.join(dpath, info['dir'], info['seg'])
    print(fpath)
    seg = emio.imread(fpath).astype('uint32')
    if pad:
        seg = np.pad(seg, pad_width, 'constant')
    dset['seg'] = seg

    # Train mask
    fpath = os.path.join(dpath, info['dir'], 'msk_train.h5')
    print(fpath)
    msk = emio.imread(fpath).astype('uint8')
    if pad:
        msk = np.pad(msk, pad_width, 'constant')
    dset['msk_train'] = msk

    # Validation mask
    fpath = os.path.join(dpath, info['dir'], 'msk_val.h5')
    print(fpath)
    msk = emio.imread(fpath).astype('uint8')
    if pad:
        msk = np.pad(msk, pad_width, 'constant')
    dset['msk_val'] = msk

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
