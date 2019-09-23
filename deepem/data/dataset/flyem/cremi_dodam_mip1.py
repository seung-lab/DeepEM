import os

import dataprovider3.emio as emio


# CREMI-Dodam dataset
data_dir = 'flyem/ground_truth'
data_info = {
    'cremi_dodam_a':{
        'img': 'img.h5',
        'seg': 'seg_b0.h5',
        'msk': 'msk',
        'dir': 'cremi_dodam_a/mip1/padded_x0_y0_z0',
        'loc': True,
    },
    'cremi_dodam_b':{
        'img': 'img.h5',
        'seg': 'seg_b0.h5',
        'msk': 'msk',
        'dir': 'cremi_dodam_b/mip1/padded_x0_y0_z0',
        'loc': True,
    },
    'cremi_dodam_c':{
        'img': 'img.h5',
        'seg': 'seg_b0.h5',
        'msk': 'msk',
        'dir': 'cremi_dodam_c/mip1/padded_x0_y0_z0',
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
    dset['img']  = emio.imread(fpath).astype('float32')
    dset['img'] /= 255.0

    # Segmentation
    fpath = os.path.join(dpath, info['dir'], info['seg'])
    print(fpath)
    dset['seg'] = emio.imread(fpath).astype('uint32')

    # Mask
    fpath = os.path.join(dpath, info['dir'], info['msk'] + '_train.h5')
    print(fpath)
    dset['msk_train'] = emio.imread(fpath).astype('uint8')
    fpath = os.path.join(dpath, info['dir'], info['msk'] + '_val.h5')
    print(fpath)
    dset['msk_val'] = emio.imread(fpath).astype('uint8')    

    # Additoinal info
    dset['loc'] = info['loc']

    return dset
