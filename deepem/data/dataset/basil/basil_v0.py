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


def load_data(data_dir, data_ids=None, **kwargs):
    if data_ids is None:
        data_ids = basil_info.keys()
    data = dict()
    base = os.path.expanduser(data_dir)
    dpath = os.path.join(base, basil_dir)
    for data_id in data_ids:
        info = basil_info[data_id]
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
    fpath = os.path.join(dpath, tag, info['dir'], info['msk'])
    print(fpath)
    dset['msk'] = emio.imread(fpath).astype('uint8')

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
