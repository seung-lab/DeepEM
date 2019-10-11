import argparse
import numpy as np
from collections import namedtuple


def dict2tuple(d):
    return namedtuple('GenericDict', d.keys())(**d)


def crop_border(img, size):
    assert(all([a > b for a, b in zip(img.shape[-3:], size[-3:])]))
    sz, sy, sx = [s // 2 for s in size[-3:]]
    return img[..., sz:-sz, sy:-sy, sx:-sx]


def crop_center(img, size):
    assert(all([a >= b for a, b in zip(img.shape[-3:], size[-3:])]))
    z, y, x = size[-3:]
    sx = (img.shape[-1] - x) // 2
    sy = (img.shape[-2] - y) // 2
    sz = (img.shape[-3] - z) // 2
    return img[..., sz:sz+z, sy:sy+y, sx:sx+x]


def vec3(s):
    try:
        z, y, x = map(int, s.split(','))
        return (z,y,x)
    except:
        raise argparse.ArgumentTypeError("Vec3 must be z,y,x")


def vec3f(s):
    try:
        z, y, x = map(float, s.split(','))
        return (z,y,x)
    except:
        raise argparse.ArgumentTypeError("Vec3f must be z,y,x")


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
    assert data.ndim == 3
    return data


def to_tensor(data):
    """Ensure that data is a numpy 4D array."""
    assert isinstance(data, np.ndarray)
    if data.ndim == 2:
        data = data[np.newaxis,np.newaxis,...]
    elif data.ndim == 3:
        data = data[np.newaxis,...]
    elif data.ndim == 4:
        pass
    else:
        raise RuntimeError("data must be a numpy 4D array")
    assert data.ndim == 4
    return data


def seg2rgb(seg, border=True):
    unq, unq_inv = np.unique(seg, return_inverse=True)

    # Random colormap
    N = len(unq)
    R = np.random.rand(N)
    G = np.random.rand(N)
    B = np.random.rand(N)
    if border:
        z = (unq == 0)
        R[z] = G[z] = B[z] = 0

    R = R[unq_inv].reshape(seg.shape)
    G = G[unq_inv].reshape(seg.shape)
    B = B[unq_inv].reshape(seg.shape)
    return np.stack((R,G,B), axis=0).astype(np.float32)
