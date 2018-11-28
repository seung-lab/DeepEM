from __future__ import print_function
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from deepem.utils import py_utils


def get_pair_first(arr, edge):
    shape = arr.size()[-3:]
    edge = np.array(edge)
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)
    ret = arr[..., os1[0]:shape[0]-os2[0],
                   os1[1]:shape[1]-os2[1],
                   os1[2]:shape[2]-os2[2]]
    return ret


def get_pair(arr, edge):
    shape = arr.size()[-3:]
    edge = np.array(edge)
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)
    arr1 = arr[..., os1[0]:shape[0]-os2[0],
                    os1[1]:shape[1]-os2[1],
                    os1[2]:shape[2]-os2[2]]
    arr2 = arr[..., os2[0]:shape[0]-os1[0],
                    os2[1]:shape[1]-os1[1],
                    os2[2]:shape[2]-os1[2]]
    return arr1, arr2
