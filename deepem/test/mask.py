from __future__ import print_function
import numpy as np

"""
Adapted from https://github.com/seung-lab/chunkflow/.
"""

class PatchMask(np.ndarray):
    def __new__(cls, patch_size, overlap):
        assert len(patch_size) == 3
        assert len(overlap) == 3

        mask = make_mask(patch_size, overlap)
        return np.asarray(mask).view(cls)


class AffinityMask(np.ndarray):
    def __new__(cls, patch_size, overlap, edges, bump):
        assert len(patch_size) == 3
        assert len(overlap) == 3
        assert len(edges) > 0
        assert bump in ['zung','wu']

        masks = list()
        for edge in edges:
            mask = make_mask(patch_size, overlap, edge=edge, bump=bump)
            masks.append(mask)
        mask = np.stack(masks)
        return np.asarray(mask).view(cls)


def make_mask(patch_size, overlap, edge=None, bump='zung'):
    # Stride
    stride = tuple(p - o for p, o in zip(patch_size, overlap))

    # Offsets of the 3x3x3 grid
    offsets = list()
    for z in range(3):
        for y in range(3):
            for x in range(3):
                offsets.append((z,y,x))

    # Slices
    slices = list()
    for offset in offsets:
        s = [slice(o*s,o*s+p) for o,s,p in zip(offset,stride,patch_size)]
        slices.append(s)

    # Shape of the 3x3x3 overlapping grid
    shape = tuple(f + 2*s for f, s in zip(patch_size, stride))
    base_mask = np.zeros(shape, dtype=np.float64)

    if bump == 'zung':
        # Max logit
        max_logit = np.full(shape, -np.inf, dtype=np.float64)
        logit = bump_logit_map(patch_size)
        for s in slices:
            max_logit[s] = np.maximum(max_logit[s], logit)

        # Mask
        for s in slices:
            base_mask[s] += bump_map(logit, max_logit[s], edge=edge)

        # Normalized weight
        s = [slice(s,s+p) for s,p in zip(stride,patch_size)]
        weight = bump_map(logit, max_logit[s]) / base_mask[s]

    elif bump == 'wu':
        # Mask
        bmap = bump_map2(patch_size, edge=edge)
        for s in slices:
            base_mask[s] += bmap

        # Normalized weight
        s = [slice(s,s+p) for s,p in zip(stride,patch_size)]
        weight = bmap / base_mask[s]

    else:
        assert False

    return np.asarray(weight, dtype=np.float32)


def bump_logit(z, y, x, t=1.5):
    return -(x*(1-x))**(-t)-(y*(1-y))**(-t)-(z*(1-z))**(-t)


def bump_logit_map(patch_size):
    x = range(patch_size[-1])
    y = range(patch_size[-2])
    z = range(patch_size[-3])
    zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
    xv = (xv + 1.0)/(patch_size[-1] + 1.0)
    yv = (yv + 1.0)/(patch_size[-2] + 1.0)
    zv = (zv + 1.0)/(patch_size[-3] + 1.0)
    return np.asarray(bump_logit(zv, yv, xv), dtype=np.float64)


def mask_edge(weight, edge=None):
    if edge is not None:
        assert len(edge) == 3
        z,y,x = edge
        assert abs(x) < weight.shape[-1]
        if x > 0:
            weight[:,:,:x] = 0
        elif x < 0:
            weight[:,:,x:] = 0
        assert abs(y) < weight.shape[-2]
        if y > 0:
            weight[:,:y,:] = 0
        elif y < 0:
            weight[:,y:,:] = 0
        assert abs(z) < weight.shape[-3]
        if z > 0:
            weight[:z,:,:] = 0
        elif z < 0:
            weight[z:,:,:] = 0
    return weight


def bump_map(logit, max_logit, edge=None):
    weight = np.exp(logit - max_logit)
    return mask_edge(weight, edge=edge)


def bump_map2(patch_size, edge=None):
    x = range(patch_size[-1])
    y = range(patch_size[-2])
    z = range(patch_size[-3])
    zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')
    xv = (xv+1.0)/(patch_size[-1]+1.0) * 2.0 - 1.0
    yv = (yv+1.0)/(patch_size[-2]+1.0) * 2.0 - 1.0
    zv = (zv+1.0)/(patch_size[-3]+1.0) * 2.0 - 1.0
    weight = np.exp(-1.0/(1.0-xv*xv) +
                    -1.0/(1.0-yv*yv) +
                    -1.0/(1.0-zv*zv))
    weight = mask_edge(weight, edge=edge)
    return np.asarray(weight, dtype=np.float64)
