from __future__ import print_function
import os

import dataprovider3.emio as emio


pinky_dir = 'pinky/ground_truth'
pinky_info = {
    'stitched_vol19-vol34':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'stitched_vol19-vol34/padded_z32_y512_x512',
        'loc': True,
    },
    'stitched_vol40-vol41':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'stitched_vol40-vol41/padded_z32_y512_x512',
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
        data[data_id] = load_dataset(dpath, data_id, info, **kwargs)
    return data


def load_dataset(dpath, tag, info, class_keys=[], **kwargs):
    assert len(class_keys) > 0
    dset = dict()

    # Image
    fpath = os.path.join(dpath, info['dir'], info['img'])
    print(fpath)
    dset['img'] = emio.imread(fpath).astype('float32') / 255.0

    # Mask
    if tag == 'stitched_vol19-vol34':
        # Train
        fpath = os.path.join(dpath, info['dir'], 'msk_train.h5')
        print(fpath)
        dset['msk_train'] = emio.imread(fpath).astype('uint8')
        # Validation
        fpath = os.path.join(dpath, info['dir'], 'msk_val.h5')
        print(fpath)
        dset['msk_val'] = emio.imread(fpath).astype('uint8')
    else:
        fpath = os.path.join(dpath, info['dir'], info['msk'])
        print(fpath)
        dset['msk'] = emio.imread(fpath).astype('uint8')

    # Segmentation
    if 'aff' in class_keys or 'long' in class_keys:
        fpath = os.path.join(dpath, info['dir'], info['seg'])
        print(fpath)
        dset['seg'] = emio.imread(fpath).astype('uint32')

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
