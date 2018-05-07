from __future__ import print_function
import os

import dataprovider3.emio as emio


# Pinky dataset
pinky_dir = 'pinky/ground_truth'
pinky_info = {
    'stitched_vol19-vol34':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'stitched_vol19-vol34',
        'loc': True,
    },
}


def load_data(data_dir, data_ids=None, **kwargs):
    if data_ids is None:
        data_ids = pinky_info.keys()
    data = dict()
    base = os.path.expanduser(data_dir)
    dpath = os.path.join(base, pinky_dir)
    for data_id in data_ids:
        info = pinky_info[data_id]
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
    if tag == 'stitched_vol19-vol34':
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
