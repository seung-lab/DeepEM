from __future__ import print_function
import os

import dataprovider3.emio as emio


# Basil dataset
basil_dir = 'basil/ground_truth'
basil_info = {
    'vol001':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.d128.h5',
        'dir': 'mip0/padded_x512_y512_z32',
        'loc': True,
    },
    'vol002':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.d128.h5',
        'dir': 'mip0/padded_x512_y512_z32',
        'loc': True,
    },
    'vol003':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.h5',
        'dir': 'mip0/padded_x512_y512_z32',
        'loc': True,
    },
    'vol004':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.d40.h5',
        'dir': 'mip0/padded_x512_y512_z32',
        'loc': True,
    },
    'vol005':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.h5',
        'dir': 'mip0/padded_x512_y512_z32',
        'loc': True,
    },
    'vol006':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.d50.h5',
        'dir': 'mip0/padded_x512_y512_z32',
        'loc': True,
    },
    'vol008':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.d10.h5',
        'dir': 'mip0/padded_x512_y512_z32',
        'loc': True,
    },
    'vol011':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.h5',
        'dir': 'mip0/padded_x512_y512_z32',
        'loc': True,
    },
}


# Pinky dataset
pinky_dir = 'pinky/ground_truth'
pinky_info = {
    'stitched_vol19-vol34':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'padded_z32_y512_x512',
        'loc': True,
    },
    'stitched_vol40-vol41':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'padded_z32_y512_x512',
        'loc': True,
    },
    'vol101':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'padded_z32_y512_x512',
        'loc': True,
    },
    'vol102':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'padded_z32_y512_x512',
        'loc': True,
    },
    'vol103':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'padded_z32_y512_x512',
        'loc': True,
    },
    'vol104':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'padded_z32_y512_x512',
        'loc': True,
    },
    'vol401':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'padded_z32_y512_x512',
        'loc': True,
    },
    'vol501':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'padded_z32_y512_x512',
        'loc': True,
    },
    'vol502':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'padded_z32_y512_x512',
        'loc': True,
    },
    'vol503':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'dir': 'padded_z32_y512_x512',
        'loc': True,
    },
}


def load_data(data_dir, data_ids=None, **kwargs):
    if data_ids is None:
        data_ids = basil_info.keys() + pinky_info.keys()

    data = dict()
    base = os.path.expanduser(data_dir)

    for data_id in data_ids:
        # Basil
        if data_id in basil_info:
            dpath = os.path.join(base, basil_dir)
            info = basil_info[data_id]
            data[data_id] = load_dataset(dpath, data_id, info)
        # Pinky
        if data_id in pinky_info:
            dpath = os.path.join(base, pinky_dir)
            info = pinky_info[data_id]
            data[data_id] = load_dataset(dpath, data_id, info)

    return data


def load_dataset(dpath, tag, info):
    dset = dict()

    # Image
    fpath = os.path.join(dpath, tag, info['dir'], info['img'])
    print(fpath)
    dset['img']  = emio.imread(fpath).astype('float32')
    dset['img'] /= 255.0

    # Segmentation
    fpath = os.path.join(dpath, tag, info['dir'], info['seg'])
    print(fpath)
    dset['seg'] = emio.imread(fpath).astype('uint32')

    # Mask
    if tag == 'stitched_vol19-vol34':
        fpath = os.path.join(dpath, tag, info['dir'], 'msk_train.h5')
        print(fpath)
        dset['msk_train'] = emio.imread(fpath).astype('uint8')
        fpath = os.path.join(dpath, tag, info['dir'], 'msk_val.h5')
        print(fpath)
        dset['msk_val'] = emio.imread(fpath).astype('uint8')
    else:
        fpath = os.path.join(dpath, tag, info['dir'], info['msk'])
        print(fpath)
        dset['msk'] = emio.imread(fpath).astype('uint8')

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
