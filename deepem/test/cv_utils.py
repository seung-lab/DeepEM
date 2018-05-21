from __future__ import print_function

import cloudvolume as cv
from taskqueue import LocalTaskQueue
import igneous
from igneous.task_creation import *

from deepem.utils import py_utils


def make_info(num_channels, layer_type, dtype, shape, resolution,
              offset=(0,0,0), chunk_size=(64,64,64)):
    return cv.CloudVolume.create_new_info(
        num_channels, layer_type, dtype, 'raw', resolution, offset, shape,
        chunk_size=chunk_size)


def ingest(data, opt):
    # Neuroglancer format
    data = py_utils.to_tensor(data)
    data = data.transpose((3,2,1,0))
    num_channels = data.shape[-1]
    shape = data.shape[:-1]
    info = make_info(num_channels, 'image', str(data.dtype), shape,
                     opt.resolution)
    print(info)
    cvol = cv.CloudVolume(opt.gs_output, mip=0, info=info,
                          parallel=opt.parallel)
    cvol[:,:,:,:] = data
    cvol.commit_info()

    # Downsample
    if opt.downsample:
        with LocalTaskQueue(parallel=opt.parallel) as tq:
            create_downsampling_tasks(tq, opt.gs_output, mip=0,
                                      fill_missing=True)
