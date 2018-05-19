from __future__ import print_function
import os

import dataprovider3.emio as emio


# Pinky dataset
pinky_dir = 'pinky/ground_truth'
pinky_info = {
    'stitched_vol19-vol34':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'stitched_vol19-vol34',
        'loc': True,
    },
    'stitched_vol40-vol41':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'stitched_vol40-vol41',
        'loc': False,
    },
    'vol101':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'vol101/padded_z32_y256_x256',
        'loc': True,
    },
    'vol102':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'vol102/padded_z32_y256_x256',
        'loc': True,
    },
    'vol103':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'vol103/padded_z32_y256_x256',
        'loc': True,
    },
    'vol104':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'vol104/padded_z32_y256_x256',
        'loc': True,
    },
    'vol401':{
        'img': 'img.h5',
        'seg': 'seg.d10.b1.h5',
        'msk': 'msk.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'vol401/myelinated',
        'loc': False,
    },
    'vol501':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk_bdr_d128.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'vol501/padded_z32_y256_x256',
        'loc': True,
    },
    'vol502':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'vol502/padded_z32_y256_x256',
        'loc': True,
    },
    'vol503':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'msk': 'msk.h5',
        'psd': 'psd.h5',
        'mit': 'mit.h5',
        'dir': 'vol503/padded_z32_y256_x256',
        'loc': True,
    },
}


def load_data(data_dir, data_ids=None, seg=True, psd=False, mit=False):
    assert any([x for x in [seg,psd,mit]])
    if data_ids is None:
        data_ids = pinky_info.keys()
    data = dict()
    base = os.path.expanduser(data_dir)
    dpath = os.path.join(base, pinky_dir)
    for data_id in data_ids:
        info = pinky_info[data_id]
        data[data_id] = load_dataset(dpath, data_id, info, seg, psd, mit)
    return data


def load_dataset(dpath, tag, info, seg, psd, mit):
    dset = dict()

    # Image
    fpath = os.path.join(dpath, info['dir'], info['img'])
    print(fpath)
    dset['img']  = emio.imread(fpath).astype('float32')
    dset['img'] /= 255.0

    # Segmentation
    if seg:
        fpath = os.path.join(dpath, info['dir'], info['seg'])
        print(fpath)
        dset['seg'] = emio.imread(fpath).astype('uint32')

    # Synapse
    if psd:
        fpath = os.path.join(dpath, info['dir'], info['psd'])
        print(fpath)
        dset['psd'] = emio.imread(fpath)
        dset['psd'] = (dset['psd'] > 0).astype('float32')

    # Mitochondria
    if mit:
        fpath = os.path.join(dpath, info['dir'], info['mit'])
        print(fpath)
        dset['mit'] = emio.imread(fpath)
        dset['mit'] = (dset['mit'] > 0).astype('float32')

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
