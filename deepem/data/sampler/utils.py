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


def recompute_CC(seg, dtype='float32'):
    """Recompute connected components"""
    seg = to_volume(seg).astype('uint32')
    aff = datatools.make_affinity(seg)
    return datatools.get_segmentation(aff).astype(dtype)
