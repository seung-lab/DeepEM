import os

import dataprovider3.emio as emio


# New CREMI
data_info = {
    'cremi_b':{
        'img': 'img.h5',
        'seg': 'seg.h5',
        'seg_d3_b0': 'seg_d3_b0.h5',
        'gli': 'glia.h5',
        'msk': 'msk.h5',
        'dir': '',
        'loc': True,
    },
}


def load_data(data_dir, data_ids=None, **kwargs):
    if data_ids is None:
        data_ids = data_info.keys()
    data = dict()
    dpath = os.path.expanduser(data_dir)
    for data_id in data_ids:
        info = data_info[data_id]
        data[data_id] = load_dataset(dpath, data_id, info, **kwargs)
    return data


def load_dataset(dpath, tag, info, class_keys=[], **kwargs):
    assert len(class_keys) > 0
    dset = dict()

    # Image
    fpath = os.path.join(dpath, info['dir'], info['img'])
    print(fpath)
    dset['img']  = emio.imread(fpath).astype('float32')
    dset['img'] /= 255.0

    # Mask
    fpath = os.path.join(dpath, info['dir'], info['msk'] + '_train.h5')
    print(fpath)
    dset['msk_train'] = emio.imread(fpath).astype('uint8')
    fpath = os.path.join(dpath, info['dir'], info['msk'] + '_val.h5')
    print(fpath)
    dset['msk_val'] = emio.imread(fpath).astype('uint8')

    # Segmentation
    if 'aff' in class_keys:
        fpath = os.path.join(dpath, info['dir'], info['seg'])
        print(fpath)
        dset['seg'] = emio.imread(fpath).astype('uint32')

    # Glia
    if 'glia' in class_keys:
        fpath = os.path.join(dpath, info['dir'], info['glia'])
        print(fpath)
        dset['glia'] = emio.imread(fpath).astype('uint8')

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
