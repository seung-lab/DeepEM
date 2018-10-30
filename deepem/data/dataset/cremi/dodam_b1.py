from __future__ import print_function
import os

import dataprovider3.emio as emio


# CREMI-Dodam dataset
dodam_dir = 'dodam/data'
cremi_info = {
    'A':{
        'img': 'A_img.h5',
        'seg': 'A_segb1.h5',
        'msk': 'A_mskb1',
        'dir': '',
        'loc': True,
    },
    'B':{
        'img': 'B_img.h5',
        'seg': 'B_segb1.h5',
        'msk': 'B_mskb1',
        'dir': '',
        'loc': True,
    },
    'C':{
        'img': 'C_img.h5',
        'seg': 'C_segb1.h5',
        'msk': 'C_mskb1',
        'dir': '',
        'loc': True,
    },
}


def load_data(data_dir, data_ids=None, **kwargs):
    if data_ids is None:
        data_ids = cremi_info.keys()
    data = dict()
    base = os.path.expanduser(data_dir)
    dpath = os.path.join(base, dodam_dir)
    for data_id in data_ids:
        info = cremi_info[data_id]
        data[data_id] = load_dataset(dpath, data_id, info)
    return data


def load_dataset(dpath, tag, info):
    dset = dict()

    # Image
    fpath = os.path.join(dpath, info['dir'], info['img'])
    print(fpath)
    dset['img']  = emio.imread(fpath).astype('float32')
    dset['img'] /= 255.0

    # Segmentation
    fpath = os.path.join(dpath, info['dir'], info['seg'])
    print(fpath)
    dset['seg'] = emio.imread(fpath).astype('uint32')

    # Mask
    fpath = os.path.join(dpath, info['dir'], info['msk'], '_train.h5')
    print(fpath)
    dset['msk_train'] = emio.imread(fpath).astype('uint8')
    fpath = os.path.join(dpath, info['dir'], info['msk'], '_val.h5')
    print(fpath)
    dset['msk_val'] = emio.imread(fpath).astype('uint8')

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
