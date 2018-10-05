from __future__ import print_function
import numpy as np
import os

import dataprovider3.emio as emio


# SNEMI3D+ dataset
data_info = {
    'train':{
        'img': 'img.h5',
        'seg': 'segmb.h5',
        'bdr': 'segb.h5',
        'msk': 'msk.h5',
        'mye': 'mye.h5',
        'dir': 'train75_val25/mip0/padded_x0_y0_z0/train',
        'loc': False,
    },
    'val':{
        'img': 'img.h5',
        'seg': 'segmb.h5',
        'bdr': 'segb.h5',
        'msk': 'msk.h5',
        'mye': 'mye.h5',
        'dir': 'train75_val25/mip0/padded_x0_y0_z0/val',
        'loc': False,
    },
}


def load_data(data_dir, data_ids=None, bdr=False, mye=False, **kwargs):
    if data_ids is None:
        data_ids = data_info.keys()
    data = dict()
    dpath = os.path.expanduser(data_dir)
    for data_id in data_ids:
        info = data_info[data_id]
        data[data_id] = load_dataset(dpath, data_id, info, bdr=bdr, mye=mye)
    return data


def load_dataset(dpath, tag, info, bdr=False, mye=False):
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

    # Boundary
    if bdr:
        fpath = os.path.join(dpath, info['dir'], info['bdr'])
        print(fpath)
        bdr = emio.imread(fpath)
        dset['bdr'] = (seg != 0).astype('uint8')

    # Myelin
    if mye:
        fpath = os.path.join(dpath, info['dir'], info['mye'])
        print(fpath)
        mye = emio.imread(fpath).astype('uint8')
        dset['mye'] = mye

    # Train mask
    fpath = os.path.join(dpath, info['dir'], info['msk'])
    print(fpath)
    dset['msk'] = emio.imread(fpath).astype('uint8')

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
