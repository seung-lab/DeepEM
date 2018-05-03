from __future__ import print_function
import argparse
import os


class Options(object):
    """
    Test options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name', required=True)
        self.parser.add_argument('--model',    required=True)

        # Data
        self.parser.add_argument('--data_dir', default="")
        self.parser.add_argument('--data_names', nargs='+')
        self.parser.add_argument('--input_name', default="img.h5")

        # cuDNN auto-tuning
        self.parser.add_argument('--autotune', action='store_true')

        # Inference
        self.parser.add_argument('--gpu_id', type=str, default='0')
        self.parser.add_argument('--chkpt_num', type=int, default=0)
        self.parser.add_argument('--no_eval', action='store_true')
        self.parser.add_argument('--pretrain', action='store_true')

        # Model
        self.parser.add_argument('--fov', type=int, default=[20,256,256], nargs='+')
        self.parser.add_argument('--depth', type=int, default=4)
        self.parser.add_argument('--group', type=int, default=0)

        # Multiclass detection
        self.parser.add_argument('--aff', type=int, default=0)
        self.parser.add_argument('--psd', action='store_true')
        self.parser.add_argument('--mit', action='store_true')

        # Forward scanning
        self.parser.add_argument('--out_prefix', default='')
        self.parser.add_argument('--out_tag', default='')
        self.parser.add_argument('--overlap', type=float, default=[0.5,0.5,0.5], nargs='+')
        self.parser.add_argument('--crop', type=int, default=None, nargs='+')

        # Benchmark
        self.parser.add_argument('--dummy', action='store_true')
        self.parser.add_argument('--input_size', type=int, default=[128,1024,1024], nargs='+')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # Directories
        opt.exp_dir = 'experiments/{}'.format(opt.exp_name)
        opt.model_dir = os.path.join(opt.exp_dir, 'models')
        opt.fwd_dir = os.path.join(opt.exp_dir, 'forward')

        # Model spec
        opt.fov = tuple(opt.fov)
        opt.in_spec = dict(input=(1,) + opt.fov)
        opt.out_spec = dict()
        if opt.aff > 0:
            opt.out_spec['affinity'] = (opt.aff,) + opt.fov
        if opt.psd:
            opt.out_spec['synapse'] = (1,) + opt.fov
        if opt.mit:
            opt.out_spec['mitochondria'] = (1,) + opt.fov
        assert(len(opt.out_spec) > 0)

        # Scan spec
        opt.scan_spec = dict()
        if opt.aff > 0:
            opt.scan_spec['affinity'] = (3,) + opt.fov
        if opt.psd:
            opt.scan_spec['synapse'] = (1,) + opt.fov
        if opt.mit:
            opt.scan_spec['mitochondria'] = (1,) + opt.fov
        stride = self.get_stride(opt.fov, opt.overlap)
        opt.scan_params = dict(stride=stride, blend='bump')

        args = vars(opt)
        print('------------ Options -------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        return self.opt

    def get_stride(self, fov, overlap):
        assert(len(fov) == 3)
        assert(len(overlap) == 3)
        inverse = lambda f,o: float(1-o) if o>0 and o<1 else int(f-o)
        return tuple(inverse(f,o) for f,o in zip(fov,overlap))
