import os
import numpy as np

import dataprovider3.emio as emio


# Focused annotation
data_dir = 'flyem/ground_truth/focused_annotation'
data_info = {
    'vol001':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'glia': 'glia.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'dir': 'vol001/mip1/padded_x512_y512_z20',
        'lamellae': [287],
        'loc': True,
    },
    'vol002':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'mit': 'mit.h5',
        'dir': 'vol002/mip1/padded_x512_y512_z20',
        'loc': True,
    },
    'vol003':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'glia': 'glia.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'mit': 'mit.h5',
        'dir': 'vol003/mip1/padded_x512_y512_z20',
        'trachea': [16],
        'glia_msk': [78,79,80,84,85,87,89,93,97,101],
        'loc': True,
    },
    'vol004':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'glia': 'glia.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'dir': 'vol004/mip1/padded_x512_y512_z20',
        'rosetta': [1],
        'trachea': [26],
        'loc': True,
    },
    'vol005':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'glia': 'glia.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'mit': 'mit.h5',
        'dir': 'vol005/mip1/padded_x512_y512_z20',
        'esophagus': [1],
        'trachea': [10],
        'loc': True,
    },
    'vol006':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'mit': 'mit.h5',
        'dir': 'vol006/mip1/padded_x512_y512_z20',
        'glia_msk': [108,122],
        'loc': True,
    },
    'vol007':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'glia': 'glia.h5',
        'mit': 'mit.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'dir': 'vol007/mip1/padded_x512_y512_z20',
        'loc': True,
    },
    'vol008':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'glia': 'glia.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'mit': 'mit.h5',
        'dir': 'vol008/mip1/padded_x512_y512_z20',
        'trachea': [67],
        'glia_msk': [62,74,75,76,77,79,80],
        'loc': True,
    },
    'vol009':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'glia': 'glia.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'mit': 'mit.h5',
        'dir': 'vol009/mip1/padded_x512_y512_z20',
        'glia_msk': [52],
        'loc': True,
    },
    'vol010':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'glia': 'glia.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'mit': 'mit.h5',
        'dir': 'vol010/mip1/padded_x512_y512_z20',
        'glia_msk': [5,18,20],
        'dark_cell': [1],
        'loc': True,
    },
    'vol011':{
        'img': 'img.h5',
        'msk': 'msk.h5',
        'seg': 'seg.h5',
        'glia': 'glia.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'mit': 'mit.h5',
        'dir': 'vol011/mip1/padded_x512_y512_z20',
        'lamellae': [52],
        'trachea': [9],
        'glia_msk': [37,38,40,42,43],
        'loc': True,
    },
}


def load_data(base_dir, data_ids=None, **kwargs):
    if data_ids is None:
        data_ids = data_info.keys()
    data = dict()
    base = os.path.expanduser(base_dir)
    dpath = os.path.join(base, data_dir)
    for data_id in data_ids:
        if data_id in data_info:
            info = data_info[data_id]
            data[data_id] = load_dataset(dpath, data_id, info, **kwargs)
    return data


def load_dataset(dpath, tag, info, class_keys=[], **kwargs):
    assert len(class_keys) > 0
    dset = dict()

    # Image
    fpath = os.path.join(dpath, info['dir'], info['img'])
    print(fpath)
    dset['img'] = emio.imread(fpath).astype(np.float32)
    dset['img'] /= 255.0

    # Mask
    fpath = os.path.join(dpath, info['dir'], info['msk'])
    print(fpath)
    dset['msk'] = emio.imread(fpath).astype(np.uint8)

    # Mitochondria
    if 'mit' in info:
        fpath = os.path.join(dpath, info['dir'], info['mit'])
        print(fpath)
        dset['mit'] = (emio.imread(fpath) > 0).astype(np.uint8)
    else:
        dset['mit'] = np.zeros_like(dset['msk'])

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
