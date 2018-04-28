from __future__ import print_function
import numpy as np

import datatools


def to_volume(data):
    """Ensure that data is a numpy 3D array."""
    assert isinstance(data, np.ndarray)
    if data.ndim == 2:
        data = data[np.newaxis,...]
    elif data.ndim == 3:
        pass
    elif data.ndim == 4:
        assert data.shape[0]==1
        data = np.squeeze(data, axis=0)
    else:
        raise RuntimeError("data must be a numpy 3D array")
    assert data.ndim==3
    return data


def affinitize(img, ret=None, edge=(1,1,1), dtype='float32'):
    """Transform segmentation to an affinity map.

    Args:
        img (ndarray): 3D indexed image, with each index corresponding to
                       each segment.

    Returns:
        ndarray: affinity map.
    """
    img = to_volume(img)
    if ret is None:
        ret = np.zeros(img.shape, dtype=dtype)

    # Sanity check
    (dz,dy,dx) = edge
    assert abs(dx) < img.shape[-1]
    assert abs(dy) < img.shape[-2]
    assert abs(dz) < img.shape[-3]

    # Slices
    s0 = list()
    s1 = list()
    s2 = list()
    for i in range(3):
        if edge[i] == 0:
            s0.append(slice(None))
            s1.append(slice(None))
            s2.append(slice(None))
        elif edge[i] > 0:
            s0.append(slice(edge[i],  None))
            s1.append(slice(edge[i],  None))
            s2.append(slice(None, -edge[i]))
        else:
            s0.append(slice(None,  edge[i]))
            s1.append(slice(-edge[i], None))
            s2.append(slice(None,  edge[i]))

    ret[s0] = (img[s1]==img[s2]) & (img[s1]>0)
    return ret[np.newaxis,...]


def recompute_CC(seg, dtype='float32'):
    """Recompute connected components"""
    seg = to_volume(seg)
    shape = (3,) + seg.shape[-3:]
    aff = np.zeros(shape, dtype=dtype)
    affinitize(seg, ret=aff[0,...], edge=(0,0,1), dtype=dtype)
    affinitize(seg, ret=aff[1,...], edge=(0,1,0), dtype=dtype)
    affinitize(seg, ret=aff[2,...], edge=(1,0,0), dtype=dtype)
    return datatools.get_segmentation(aff).astype(dtype)
