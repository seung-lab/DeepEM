from __future__ import print_function
import imp
import numpy as np
import os

from dataprovider3 import Dataset, ForwardScanner, emio

from deepem.test.model import Model, OnnxModel
from deepem.utils.py_utils import crop_center


def load_model(opt):
    # Create a model.
    mod = imp.load_source('model', opt.model)
    if opt.onnx:
        model = OnnxModel(mod.create_model(opt), opt)
    else:
        model = Model(mod.create_model(opt), opt)

    # Load from a checkpoint, if any.
    if opt.chkpt_num > 0:
        model = load_chkpt(model, opt.model_dir, opt.chkpt_num)

    model = model.train() if opt.no_eval else model.eval()
    return model.cuda()


def load_chkpt(model, fpath, chkpt_num):
    print("LOAD CHECKPOINT: {} iters.".format(chkpt_num))
    fname = os.path.join(fpath, "model{}.chkpt".format(chkpt_num))
    model.load(fname)
    return model


def make_forward_scanner(data_name, opt):
    # Read an EM image.
    if opt.dummy:
        img = np.random.rand(*opt.input_size[-3:]).astype('float32')
    else:
        fpath = os.path.join(opt.data_dir, data_name, opt.input_name)
        img = emio.imread(fpath)
        img = (img/255.).astype('float32')

    # Border mirroring
    if opt.crop:
        pad_width = [(x//2,x//2) for x in opt.crop]
        img = np.pad(img, pad_width, 'reflect')

    # ForwardScanner
    dataset = Dataset(spec=opt.in_spec)
    dataset.add_data('input', img)
    return ForwardScanner(dataset, opt.scan_spec, **opt.scan_params)


def save_output(data_name, output, opt):
    for k in output.data:
        data = output.get_data(k)
        if opt.crop:
            data = crop_center(data, opt.crop)
        dname = data_name.replace('/', '_')
        fname = "{}_{}_{}".format(dname, k, opt.chkpt_num)
        if opt.out_prefix:
            fname = opt.out_prefix + '_' + fname
        if opt.out_tag:
            fname = fname + '_' + opt.out_tag
        fpath = os.path.join(opt.fwd_dir, fname + ".h5")
        emio.imsave(data, fpath)
