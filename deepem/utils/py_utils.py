from __future__ import print_function
import numpy as np
from collections import namedtuple


def dict2tuple(d):
    return namedtuple('GenericDict', d.keys())(**d)


def crop_center(img, size):
    assert(all([a >= b for a, b in zip(img.shape[-3:], size[-3:])]))
    z, y, x = size[-3:]
    sx = (img.shape[-1] - x) // 2
    sy = (img.shape[-2] - y) // 2
    sz = (img.shape[-3] - z) // 2
    return img[..., sz:sz+z, sy:sy+y, sx:sx+x]
