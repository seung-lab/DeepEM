import imp
import numpy as np
import os
from types import SimpleNamespace

from dataprovider3 import Dataset, ForwardScanner, emio

from deepem.test.model import Model
from deepem.utils import py_utils


def load_model(opt):
    # Create a model.
    mod = imp.load_source('model', opt.model)
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


def make_forward_scanner(opt, data_name=None):
    # Cloud-volume
    if opt.gs_input:
        try:
            from deepem.test import cv_utils
            img = cv_utils.cutout(opt, opt.gs_input, dtype='uint8')

            # Optional input histogram normalization 
            if opt.gs_input_norm:
                assert len(opt.gs_input_norm) == 2
                low, high = opt.gs_input_norm
                img = normalize_per_slice(img, lowerfract=low, upperfract=high)
            
            # [0, 255] -> [0.0, 1.0]
            img = (img/255.).astype('float32')

            # Optional input mask
            if opt.gs_input_mask:
                try:
                    msk = cv_utils.cutout(opt, opt.gs_input_mask, dtype='uint8')
                    img[msk > 0] = 0
                except:
                    raise

        except ImportError:
            raise
    else:
        assert data_name is not None
        print(data_name)
        # Read an EM image.
        if opt.dummy:
            img = np.random.rand(*opt.dummy_inputsz[-3:]).astype('float32')
        else:
            fpath = os.path.join(opt.data_dir, data_name, opt.input_name)
            img = emio.imread(fpath)
            img = (img/255.).astype('float32')

        # Border mirroring
        if opt.mirror:
            pad_width = [(x//2,x//2) for x in opt.mirror]
            img = np.pad(img, pad_width, 'reflect')

    # ForwardScanner
    dataset = Dataset(spec=opt.in_spec)
    dataset.add_data('input', img)
    return ForwardScanner(dataset, opt.scan_spec, **opt.scan_params)


def save_output(output, opt, data_name=None, aug_out=None):
    for k in output.data:
        data = output.get_data(k)

        # Crop
        if opt.crop_border:
            data = py_utils.crop_border(data, opt.crop_border)
        if opt.crop_center:
            data = py_utils.crop_center(data, opt.crop_center)

        # Cloud-volume
        if opt.gs_output:
            try:
                tag = k
                if opt.tags is not None:
                    if tag in opt.tags:
                        tag = opt.tags[tag]

                from deepem.test import cv_utils
                cv_utils.ingest(data, opt, tag=tag)

                # Optional variance
                if aug_out is not None:
                    variance = np.var(np.stack(aug_out[k]), axis=0)
                    cv_utils.ingest(variance, opt, tag=(tag + '_var'))

            except ImportError:
                raise
        else:
            dname = data_name.replace('/', '_')
            fname = "{}_{}_{}".format(dname, k, opt.chkpt_num)
            if opt.out_prefix:
                fname = opt.out_prefix + '_' + fname
            if opt.out_tag:
                fname = fname + '_' + opt.out_tag
            fpath = os.path.join(opt.fwd_dir, fname + ".h5")
            emio.imsave(data, fpath)


def histogram_per_slice(img):    
    z = img.shape[-3]
    xy = img.shape[-2] * img.shape[-1]
    return np.apply_along_axis(np.bincount, axis=1, arr=img.reshape((z,xy)),
                               minlength=255)


def find_section_clamping_values(zlevel, lowerfract, upperfract):
    """Find int8 values that correspond to lowerfract & upperfract of zlevel histogram
    
    From igneous (https://github.com/seung-lab/igneous/blob/master/igneous/tasks/tasks.py#L547)
    """
    filtered = np.copy(zlevel)

    # remove pure black from frequency counts as
    # it has no information in our images
    filtered[0] = 0

    cdf = np.zeros(shape=(len(filtered),), dtype=np.uint64)
    cdf[0] = filtered[0]
    for i in range(1, len(filtered)):
        cdf[i] = cdf[i - 1] + filtered[i]

    total = cdf[-1]

    if total == 0:
        return (0, 0)

    lower = 0
    for i, val in enumerate(cdf):
        if float(val) / float(total) > lowerfract:
            break
        lower = i

    upper = 0
    for i, val in enumerate(cdf):
        if float(val) / float(total) > upperfract:
            break
        upper = i

    return (lower, upper)


def normalize_per_slice(img, lowerfract=0.01, upperfract=0.01):
    maxval = 255.
    hist = histogram_per_slice(img)
    img = img.astype(np.float32)
    for z in range(img.shape[-3]):
        lower, upper = find_section_clamping_values(hist[z], 
                                                lowerfract=lowerfract, 
                                                upperfract=1-upperfract)
        if lower == upper:
            continue

        im = img[z,:,:]
        im = (im - float(lower)) * (maxval / (float(upper) - float(lower)))
        img[z,:,:] = im

    img = np.round(img)
    return np.clip(img, 0., maxval).astype(np.uint8)