from __future__ import print_function
import os

import dataprovider3.emio as emio


# SNEMI3D dataset
snemi3d_info = {
    'train':{
        'img': 'train_img.h5',
        'seg': 'train_seg',
        'msk': None,
        'dir': '',
        'loc': True,
    },
}


def load_data(data_dir, data_ids=None, **kwargs):
    if data_ids is None:
        data_ids = snemi3d_info.keys()
    data = dict()
    base = os.path.expanduser(data_dir)
    dpath = os.path.join(base, pinky_dir)
    for data_id in data_ids:
        info = snemi3d_info[data_id]
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
    # Train
    fpath = os.path.join(dpath, info['dir'], 'train_msk.h5')
    print(fpath)
    dset['msk_train'] = emio.imread(fpath).astype('uint8')
    # Validation
    fpath = os.path.join(dpath, info['dir'], 'val_msk.h5')
    print(fpath)
    dset['msk_val'] = emio.imread(fpath).astype('uint8')

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
