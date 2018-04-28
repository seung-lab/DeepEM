from __future__ import print_function
import numpy as np

import torch


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


def get_pair_first2(arr, edge):
    # Margin correction (only works for (1,1,1))
    edge = np.array(list(map(lambda x: x - np.sign(x), edge)))
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)
    shape = arr.size()[-3:]
    ret = arr[..., os1[0]:shape[0]-os2[0],
                   os1[1]:shape[1]-os2[1],
                   os1[2]:shape[2]-os2[2]]
    return ret


def get_pair2(arr, edge):
    edge = np.array(edge)
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)

    shape = arr.size()[-3:]
    arr1 = arr[..., os1[0]:shape[0]-os2[0],
                    os1[1]:shape[1]-os2[1],
                    os1[2]:shape[2]-os2[2]]
    arr2 = arr[..., os2[0]:shape[0]-os1[0],
                    os2[1]:shape[1]-os1[1],
                    os2[2]:shape[2]-os1[2]]

    # Margin correction (only works for (1,1,1))
    m1 = list(map(lambda x: 0 if x > 0 else 1, edge))
    m2 = list(map(lambda x: 0 if x < 0 else 1, edge))

    shape = arr1.size()[-3:]
    arr1 = arr1[..., m1[0]:shape[0]-m2[0],
                     m1[1]:shape[1]-m2[1],
                     m1[2]:shape[2]-m2[2]]
    arr2 = arr2[..., m1[0]:shape[0]-m2[0],
                     m1[1]:shape[1]-m2[1],
                     m1[2]:shape[2]-m2[2]]

    return arr1, arr2
