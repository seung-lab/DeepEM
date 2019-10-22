import os
import sys
import datetime
from collections import OrderedDict
import numpy as np

import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from deepem.utils import torch_utils, py_utils


class Logger(object):
    def __init__(self, opt):
        self.monitor = {'train': Logger.Monitor(), 'test': Logger.Monitor()}
        self.log_dir = opt.log_dir
        self.writer = SummaryWriter(opt.log_dir)
        self.in_spec = dict(opt.in_spec)
        self.out_spec = dict(opt.out_spec)
        self.outputsz = opt.outputsz
        self.lr = opt.lr

        # Basic logging
        self.timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.log_params(vars(opt))
        self.log_command()
        self.log_command_args()

        # Blood vessel
        self.blv_num_channels = opt.blv_num_channels

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.writer:
            self.writer.close()

    def record(self, phase, loss, nmsk, **kwargs):
        monitor = self.monitor[phase]

        # Reduce to scalar values.
        to_scalar = lambda x: x.item() if torch.is_tensor(x) else x
        for k in sorted(loss):
            monitor.add_to('vals', k, to_scalar(loss[k]))
            monitor.add_to('norm', k, to_scalar(nmsk[k]))

        for k, v in kwargs.items():
            monitor.add_to('vals', k, v)
            monitor.add_to('norm', k, 1)

    def check(self, phase, iter_num):
        stats = self.monitor[phase].flush()
        self.log(phase, iter_num, stats)
        self.display(phase, iter_num, stats)

    def log(self, phase, iter_num, stats):
        for k, v in stats.items():
            self.writer.add_scalar('{}/{}'.format(phase, k), v, iter_num)

    def display(self, phase, iter_num, stats):
        disp = "[%s] Iter: %8d, " % (phase, iter_num)
        for k, v in stats.items():
            disp += "%s = %.3f, " % (k, v)
        disp += "(lr = %.6f). " % self.lr
        print(disp)

    class Monitor(object):
        def __init__(self):
            self.vals = OrderedDict()
            self.norm = OrderedDict()

        def add_to(self, name, k, v):
            assert(name in ['vals','norm'])
            d = getattr(self, name)
            if k in d:
                d[k] += v
            else:
                d[k] = v

        def flush(self):
            ret = OrderedDict()
            for k in self.vals:
                ret[k] = self.vals[k]/self.norm[k]
            self.vals = OrderedDict()
            self.norm = OrderedDict()
            return ret

    def log_images(self, phase, iter_num, preds, sample):
        # Peep output size
        key = sorted(self.out_spec)[0]
        cropsz = sample[key].shape[-3:]
        for k in sorted(self.out_spec):
            outsz = sample[k].shape[-3:]
            assert np.array_equal(outsz, cropsz)

        # Inputs
        for k in sorted(self.in_spec):
            tag = '{}/images/{}'.format(phase, k)
            tensor = sample[k][0,...].cpu()
            tensor = torch_utils.crop_center(tensor, cropsz)
            self.log_image(tag, tensor, iter_num)

        # Outputs
        for k in sorted(self.out_spec):
            if k == 'affinity':
                # Prediction
                tag = '{}/images/{}'.format(phase, k)
                tensor = torch.sigmoid(preds[k][0,0:3,...]).cpu()
                self.log_image(tag, tensor, iter_num)

                # Mask
                tag = '{}/masks/{}'.format(phase, k)
                msk = sample[k + '_mask'][0,...].cpu()
                self.log_image(tag, msk, iter_num)

                # Target
                tag = '{}/labels/{}'.format(phase, k)
                seg = sample[k][0,0,...].cpu().numpy().astype('uint32')
                rgb = torch.from_numpy(py_utils.seg2rgb(seg))
                self.log_image(tag, rgb, iter_num)

            elif k == 'myelin':
                # Prediction
                tag = '{}/images/{}'.format(phase, k)
                pred = torch.sigmoid(preds[k][0,...]).cpu()
                self.log_image(tag, pred, iter_num)

            elif k == 'synapse':
                # Prediction
                tag = '{}/images/{}'.format(phase, k)
                pred = torch.sigmoid(preds[k][0,...]).cpu()
                self.log_image(tag, pred, iter_num)

                # Target
                tag = '{}/labels/{}'.format(phase, k)
                target = sample[k][0,...].cpu()
                self.log_image(tag, target, iter_num)

            elif k == 'blood_vessel':
                # Prediction
                tag = '{}/images/{}'.format(phase, k)
                pred = torch.sigmoid(preds[k][0,...]).cpu()
                if self.blv_num_channels == 2:
                    zero = torch.zeros_like(pred[[0],...])
                    pred = torch.cat((pred,zero), dim=-4)
                self.log_image(tag, pred, iter_num)

            elif k == 'glia':
                # Prediction
                tag = '{}/images/{}'.format(phase, k)
                pred = torch.sigmoid(preds[k][0,...]).cpu()
                self.log_image(tag, pred, iter_num)

                # Target
                tag = '{}/labels/{}'.format(phase, k)
                target = sample[k][0,...].cpu()
                self.log_image(tag, target, iter_num)

    def log_image(self, tag, tensor, iter_num):
        assert(torch.is_tensor(tensor))
        depth = tensor.shape[-3]
        imgs = [tensor[:,z,:,:] for z in range(depth)]
        img = make_grid(imgs, nrow=depth, padding=0)
        self.writer.add_image(tag, img, iter_num)

    def log_params(self, params):
        fname = os.path.join(self.log_dir, "{}_params.csv".format(self.timestamp))
        with open(fname, "w+") as f:
            for k, v in params.items():
                f.write("{k}: {v}\n".format(k=k, v=v))

    def log_command(self):
        fname = os.path.join(self.log_dir, "{}_command".format(self.timestamp))
        command = " ".join(sys.argv)
        with open(fname, "w+") as f:
            f.write(command)

    def log_command_args(self):
        fname = os.path.join(self.log_dir, "{}_args.txt".format(self.timestamp))
        with open(fname, "w+") as f:
            for arg in sys.argv[1:]:
                f.write(arg + "\n")
