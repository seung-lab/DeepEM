from __future__ import print_function
import argparse
import imp
import os


class TrainOptions(object):
    """
    Training options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name', required=True)
        self.parser.add_argument('--data_dir', required=True)
        self.parser.add_argument('--data',     required=True)
        self.parser.add_argument('--model',    required=True)
        self.parser.add_argument('--augment',  required=True)
        self.parser.add_argument('--sampler',  required=True)

        # cuDNN auto-tuning
        self.parser.add_argument('--autotune', action='store_true')

        # Training/validation sets
        self.parser.add_argument('--train_ids', type=str, default=[], nargs='+')
        self.parser.add_argument('--val_ids', type=str, default=[], nargs='+')

        # Training
        self.parser.add_argument('--base_lr', type=float, default=0.001)
        self.parser.add_argument('--max_iter', type=int, default=1000000)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--num_workers', type=int, default=1)
        self.parser.add_argument('--gpu_ids', type=str, default=['0'], nargs='+')
        self.parser.add_argument('--eval_intv', type=int, default=1000)
        self.parser.add_argument('--eval_iter', type=int, default=100)
        self.parser.add_argument('--avgs_intv', type=int, default=100)
        self.parser.add_argument('--imgs_intv', type=int, default=1000)
        self.parser.add_argument('--warm_up', type=int, default=100)
        self.parser.add_argument('--chkpt_intv', type=int, default=10000)
        self.parser.add_argument('--chkpt_num', type=int, default=0)
        self.parser.add_argument('--no_eval', action='store_true')

        # Loss
        self.parser.add_argument('--size_average', action='store_true')
        self.parser.add_argument('--margin', type=float, default=0)

        # Model
        self.parser.add_argument('--fov', type=int, default=[20,256,256], nargs='+')
        self.parser.add_argument('--depth', type=int, default=4)
        self.parser.add_argument('--long_range', action='store_true')
        self.parser.add_argument('--symmetric', action='store_true')

        # Data augmentation
        self.parser.add_argument('--box', default=None)

        # Multiclass detection
        self.parser.add_argument('--aff', type=float, default=0)
        self.parser.add_argument('--psd', type=float, default=0)
        self.parser.add_argument('--mit', type=float, default=0)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # Directories
        opt.exp_dir = 'experiments/{}'.format(opt.exp_name)
        opt.log_dir = os.path.join(opt.exp_dir, 'logs')
        opt.model_dir = os.path.join(opt.exp_dir, 'models')

        # Training/validation sets
        mod = imp.load_source('sampler', opt.sampler)
        if not opt.train_ids:
            opt.train_ids = mod.get_data_ids(True)
        if not opt.val_ids:
            opt.val_ids = mod.get_data_ids(False)

        # Model spec
        opt.fov = tuple(opt.fov)
        opt.in_spec = dict(input=(1,) + opt.fov)
        opt.edges = self.get_edges(opt)
        opt.out_spec = dict()
        opt.loss_weight = dict()

        if opt.aff > 0:
            opt.out_spec['affinity'] = (len(opt.edges),) + opt.fov
            opt.loss_weight['affinity'] = opt.aff

        if opt.psd > 0:
            opt.out_spec['synapse'] = (1,) + opt.fov
            opt.loss_weight['synapse'] = opt.psd

        if opt.mit > 0:
            opt.out_spec['mitochondria'] = (1,) + opt.fov
            opt.loss_weight['mitochondria'] = opt.mit

        assert len(opt.out_spec) > 0
        assert len(opt.out_spec) == len(opt.loss_weight)

        args = vars(opt)
        print('------------ Options -------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        return self.opt

    def get_edges(self, opt):
        edges = list()

        # Nearest neighbor edges
        edges.append((0,0,1))
        edges.append((0,1,0))
        edges.append((1,0,0))

        if opt.long_range:
            # x-affinity.
            edges.append((0,0,4))
            edges.append((0,0,8))
            edges.append((0,0,12))
            edges.append((0,0,16))
            edges.append((0,0,32))
            # y-affinity.
            edges.append((0,4,0))
            edges.append((0,8,0))
            edges.append((0,12,0))
            edges.append((0,16,0))
            edges.append((0,32,0))
            # z-affinity.
            edges.append((2,0,0))
            edges.append((3,0,0))
            edges.append((4,0,0))

        if opt.symmetric:
            edges += [tuple(-x for x in e) for e in edges]

        return edges
