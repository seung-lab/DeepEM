from __future__ import print_function

import cloudvolume as cv
from cloudvolume.lib import Vec, Bbox
from taskqueue import LocalTaskQueue
import igneous
from igneous.task_creation import *

from deepem.utils import py_utils


def make_info(num_channels, layer_type, dtype, shape, resolution,
              offset=(0,0,0), chunk_size=(64,64,64)):
    return cv.CloudVolume.create_new_info(
        num_channels, layer_type, dtype, 'raw', resolution, offset, shape,
        chunk_size=chunk_size)


def cutout(opt, dtype='uint8'):
    print(opt.gs_input)

    # CloudVolume.
    cvol = cv.CloudVolume(opt.gs_input, mip=opt.in_mip, cache=opt.cache,
                          fill_missing=True, parallel=opt.parallel)

    # Cutout
    if not opt.end:
        assert opt.size is not None
        opt.end = tuple(x + y for x, y in zip(opt.begin, opt.size))
    sl = [slice(x,y) for x, y in zip(opt.begin, opt.end)]
    print('begin = {}'.format(opt.begin))
    print('end = {}'.format(opt.end))

    # Coordinates
    print('mip 0 = {}'.format(sl))
    sl = cvol.slices_from_global_coords(sl)
    print('mip {} = {}'.format(opt.in_mip, sl))
    cutout = cvol[sl]

    # Transpose & squeeze
    cutout = cutout.transpose([3,2,1,0])
    cutout = np.squeeze(cutout).astype(dtype)
    return cutout


def ingest(data, opt):
    # Neuroglancer format
    data = py_utils.to_tensor(data)
    data = data.transpose((3,2,1,0))
    num_channels = data.shape[-1]
    shape = data.shape[:-1]
    info = make_info(num_channels, 'image', str(data.dtype), shape,
                     opt.resolution, offset=opt.offset,
                     chunk_size=opt.chunk_size)
    print(info)
    if '{}' in opt.gs_output:
        coords = '_'.join(['{}-{}'.format(b,e) for b,e in zip(opt.begin,opt.end)])
        gs_path = opt.gs_output.format(coords)
    else:
        gs_path = opt.gs_output
    cvol = cv.CloudVolume(gs_path, mip=0, info=info,
                          parallel=opt.parallel)
    cvol[:,:,:,:] = data
    cvol.commit_info()

    # Downsample
    if opt.downsample:
        with LocalTaskQueue(parallel=opt.parallel) as tq:
            create_downsampling_tasks(tq, gs_path, mip=0,
                                      fill_missing=True)
