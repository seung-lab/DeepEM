from __future__ import print_function
from collections import OrderedDict

import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from deepem.utils import torch_utils


class Logger(object):
    def __init__(self, opt):
        self.monitor = {'train': Logger.Monitor(), 'test': Logger.Monitor()}
        self.writer = SummaryWriter(opt.log_dir)
        self.in_spec = dict(opt.in_spec)
        self.out_spec = dict(opt.out_spec)
        self.base_lr = opt.base_lr

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
        disp += "(lr = %.6f). " % self.base_lr
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
        # Inputs
        for k in sorted(self.in_spec):
            tag = '{}/images/{}'.format(phase, k)
            tensor = sample[k][0,...]
            self.log_image(tag, tensor, iter_num)

        # Outputs
        for k  in sorted(self.out_spec):
            if k == 'affinity':
                # Prediction
                tag = '{}/images/{}'.format(phase, k)
                tensor = F.sigmoid(preds[k][0,0:3,...])
                self.log_image(tag, tensor, iter_num)
            elif k == 'synapse' or k == 'mitochondria':
                # Prediction
                tag = '{}/images/{}'.format(phase, k)
                tensor = F.sigmoid(preds[k].data[0,...])
                self.log_image(tag, tensor, iter_num)
                # Target
                tag = '{}/labels/{}'.format(phase, k)
                tensor = sample[k][0,...]
                self.log_image(tag, tensor, iter_num)
            elif k == 'embedding':
                vec = preds[k][0,...]

                # x-affinity
                tag = '{}/images/x-affinity'.format(phase)
                aff = torch_utils.affinity(*(torch_utils.get_pair(vec, (0,0,1))))
                self.log_image(tag, aff, iter_num)

                # y-affinity
                tag = '{}/images/y-affinity'.format(phase)
                aff = torch_utils.affinity(*(torch_utils.get_pair(vec, (0,1,0))))
                self.log_image(tag, aff, iter_num)

                # z-affinity
                tag = '{}/images/z-affinity'.format(phase)
                aff = torch_utils.affinity(*(torch_utils.get_pair(vec, (1,0,0))))
                self.log_image(tag, aff, iter_num)

                # Embedding
                tag = '{}/images/embedding'.format(phase)
                vec = vec - torch.min(vec)
                vec = vec / torch.max(vec)
                self.log_image(tag, vec[0:3,...], iter_num)

    def log_image(self, tag, tensor, iter_num):
        assert(torch.is_tensor(tensor))
        depth = tensor.shape[-3]
        imgs = [tensor[:,z,:,:] for z in range(depth)]
        img = make_grid(imgs, nrow=depth, padding=0)
        self.writer.add_image(tag, img, iter_num)
