from __future__ import print_function
import argparse
import os

from deepem.utils.py_utils import vec3


class Options(object):
    """
    Training options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name', required=True)
        self.parser.add_argument('--data_dir', required=True)
        self.parser.add_argument('--data',     required=True)
        self.parser.add_argument('--model',    required=True)
        self.parser.add_argument('--sampler',  required=True)
        self.parser.add_argument('--augment',  default=None)

        # cuDNN auto-tuning
        self.parser.add_argument('--no_autotune', action='store_false')

        # Training/validation sets
        self.parser.add_argument('--train_ids', type=str, default=[], nargs='+')
        self.parser.add_argument('--train_prob', type=float, default=None, nargs='+')
        self.parser.add_argument('--val_ids', type=str, default=[], nargs='+')
        self.parser.add_argument('--val_prob', type=float, default=None, nargs='+')

        # Training
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
        self.parser.add_argument('--pretrain', default=None)

        # Loss
        self.parser.add_argument('--loss', default='BCELoss')
        self.parser.add_argument('--size_average', action='store_true')
        self.parser.add_argument('--margin0', type=float, default=0)
        self.parser.add_argument('--margin1', type=float, default=0)
        self.parser.add_argument('--inverse', action='store_true')
        self.parser.add_argument('--class_balancing', action='store_true')

        # Optimizer
        self.parser.add_argument('--optim', default='Adam')
        self.parser.add_argument('--lr', type=float, default=0.001)

        # Optimizer: Adam
        self.parser.add_argument('--betas', type=float, default=[0.9,0.999], nargs='+')
        self.parser.add_argument('--eps', type=float, default=1e-08)
        self.parser.add_argument('--amsgrad', action='store_true')

        # Optimizer: SGD
        self.parser.add_argument('--momentum', type=float, default=0.9)

        # Model architecture
        self.parser.add_argument('--inputsz', type=int, default=None, nargs='+')
        self.parser.add_argument('--outputsz', type=int, default=None, nargs='+')
        self.parser.add_argument('--fov', type=vec3, default=(20,256,256))
        self.parser.add_argument('--depth', type=int, default=4)
        self.parser.add_argument('--width', type=int, default=None, nargs='+')
        self.parser.add_argument('--group', type=int, default=0)
        self.parser.add_argument('--act', default='ReLU')

        # Data augmentation
        self.parser.add_argument('--recompute', action='store_true')
        self.parser.add_argument('--flip', action='store_true')
        self.parser.add_argument('--grayscale', action='store_true')
        self.parser.add_argument('--warping', action='store_true')
        self.parser.add_argument('--misalign', action='store_true')
        self.parser.add_argument('--interp', action='store_true')
        self.parser.add_argument('--missing', type=int, default=0)
        self.parser.add_argument('--blur', type=int, default=0)
        self.parser.add_argument('--box', default=None)
        self.parser.add_argument('--lost', action='store_true')
        self.parser.add_argument('--random', action='store_true')

        # Long-range affinity
        self.parser.add_argument('--long', type=float, default=0)
        self.parser.add_argument('--edges', type=vec3, default=[], nargs='+')

        # Multiclass detection
        self.parser.add_argument('--aff', type=float, default=0)  # Affinity
        self.parser.add_argument('--bdr', type=float, default=0)  # Boundary
        self.parser.add_argument('--syn', type=float, default=0)  # Synapse
        self.parser.add_argument('--mit', type=float, default=0)  # Mitochondria
        self.parser.add_argument('--mye', type=float, default=0)  # Myelin
        self.parser.add_argument('--bld', type=float, default=0)  # Blood vessel

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # Directories
        if opt.exp_name.split('/')[0] == 'experiments':
            opt.exp_dir = opt.exp_name
        else:
            opt.exp_dir = 'experiments/{}'.format(opt.exp_name)
        opt.log_dir = os.path.join(opt.exp_dir, 'logs')
        opt.model_dir = os.path.join(opt.exp_dir, 'models')

        # Training/validation sets
        if (not opt.train_ids) or (not opt.val_ids):
            raise ValueError("Train/validation IDs unspecified")
        if opt.train_prob:
            assert len(opt.train_ids) == len(opt.train_prob)
        if opt.val_prob:
            assert len(opt.val_ids) == len(opt.val_prob)

        args = vars(opt)

        # Loss
        loss_keys = ['size_average','margin0','margin1','inverse']
        opt.loss_params = {k: args[k] for k in loss_keys}

        # Optimizer
        if opt.optim == 'Adam':
            optim_keys = ['lr','betas','eps','amsgrad']
        elif opt.optim == 'SGD':
            optim_keys = ['lr','momentum']
        else:
            optim_keys = ['lr']
        opt.optim_params = {k: args[k] for k in optim_keys}

        # Data augmentation
        aug_keys = ['recompute','flip','grayscale','warping','misalign',
                    'interp','missing','blur','box','lost','random']
        opt.aug_params = {k: args[k] for k in aug_keys}

        # Model
        opt.fov = tuple(opt.fov)
        opt.inputsz = opt.fov if opt.inputsz is None else tuple(opt.inputsz)
        opt.outputsz = opt.fov if opt.outputsz is None else tuple(opt.outputsz)
        opt.in_spec = dict(input=(1,) + opt.inputsz)
        opt.out_spec = dict()
        opt.loss_weight = dict()

        # Multiclass detection
        class_keys = list()
        class_dict = {
            'aff':  ('affinity', 3),
            'long': ('long_range', len(opt.edges)),
            'bdr':  ('boundary', 1),
            'syn':  ('synapse', 1),
            'mit':  ('mitochondria', 1),
            'mye':  ('myelin', 1),
            'bld':  ('blood_vessel', 1)
        }

        for k, v in class_dict.items():
            loss_w = args[k]
            if loss_w > 0:
                output_name, num_channels = v
                assert num_channels > 0
                opt.out_spec[output_name] = (num_channels,) + opt.outputsz
                opt.loss_weight[output_name] = loss_w
                class_keys.append(k)

        assert len(opt.out_spec) > 0
        assert len(opt.out_spec) == len(opt.loss_weight) == len(class_keys)
        opt.data_params = dict(class_keys=class_keys)

        args = vars(opt)
        print('------------ Options -------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        return self.opt
