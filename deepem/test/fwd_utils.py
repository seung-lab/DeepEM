from __future__ import print_function
import numpy as np

from deepem.utils import py_utils


class Flip(object):
    def __call__(self, data, rule):
        """Flip data according to a specified rule.

        Args:
            data:   4D numpy array to be transformed.
            rule:   Transform rule, specified as a Boolean array.
                    [z-flip, y-flip, x-flip, xy-transpose]

        Returns:
            Transformed data.
        """
        data = py_utils.to_tensor(data)
        assert np.size(rule)==4

        # z-flip
        if rule[0]:
            data = np.flip(data, axis=-3)
        # y-flip
        if rule[1]:
            data = np.flip(data, axis=-2)
        # x-flip
        if rule[2]:
            data = np.flip(data, axis=-1)
        # xy-transpose
        if rule[3]:
            data = data.transpose(0,1,3,2)

        # Prevent potential negative stride issues by copying.
        return np.copy(data)

flip = Flip()


def revert_flip(data, rule, dst=None):
    data = py_utils.to_tensor(data)
    assert np.size(rule)==4

    # Special treat for affinity.
    is_affinity = dst is not None
    if is_affinity:
        (dz,dy,dx) = dst
        assert data.shape[-4] >= 3
        assert dx and abs(dx) < data.shape[-1]
        assert dy and abs(dy) < data.shape[-2]
        assert dz and abs(dz) < data.shape[-3]

    # xy-transpose
    if rule[3]:
        data = data.transpose(0,1,3,2)
        # Swap x/y-affinity maps.
        if is_affinity:
            data[[0,1],...] = data[[1,0],...]

    # x-flip
    if rule[2]:
        data = np.flip(data, axis=-1)
        # Special treatment for x-affinity.
        if is_affinity:
            if dx > 0:
                data[0,:,:,dx:] = data[0,:,:,:-dx]
                data[0,:,:,:dx].fill(0)
            else:
                dx = abs(dx)
                data[0,:,:,:-dx] = data[0,:,:,dx:]
                data[0,:,:,-dx:].fill(0)

    # y-flip
    if rule[1]:
        data = np.flip(data, axis=-2)
        # Special treatment for y-affinity.
        if is_affinity:
            if dy > 0:
                data[1,:,dy:,:] = data[1,:,:-dy,:]
                data[1,:,:dy,:].fill(0)
            else:
                dy = abs(dy)
                data[1,:,:-dy,:] = data[1,:,dy:,:]
                data[1,:,-dy:,:].fill(0)

    # z-flip
    if rule[0]:
        data = np.flip(data, axis=-3)
        # Special treatment for z-affinity.
        if is_affinity:
            if dz > 0:
                data[2,dz:,:,:] = data[2,:-dz,:,:]
                data[2,:dz,:,:].fill(0)
            else:
                dz = abs(dz)
                data[2,:-dz,:,:] = data[2,dz:,:,:]
                data[2,-dz:,:,:].fill(0)

    return data
