from __future__ import print_function
import argparse
import imp
import os

from deepem.utils.py_utils import vec3


class Options(object):
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
        self.parser.add_argument('--sampler',  required=True)
        self.parser.add_argument('--augment',  default=None)

        # cuDNN auto-tuning
        self.parser.add_argument('--autotune', action='store_true')

        # Training/validation sets
        self.parser.add_argument('--train_ids', type=str, default=[], nargs='+')
        self.parser.add_argument('--train_prob', type=float, default=None, nargs='+')
        self.parser.add_argument('--val_ids', type=str, default=[], nargs='+')
        self.parser.add_argument('--val_prob', type=float, default=None, nargs='+')
        self.parser.add_argument('--pad_size', type=vec3, default=(0,0,0))

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
        self.parser.add_argument('--no_logits', action='store_true')

        # Edge-based loss
        self.parser.add_argument('--max_edges', type=vec3, default=[(5,32,32)], nargs='+')
        self.parser.add_argument('--n_edge', type=int, default=32)

        # Optimizer
        self.parser.add_argument('--optim', default='Adam')
        self.parser.add_argument('--lr', type=float, default=0.001)

        # Adam
        self.parser.add_argument('--betas', type=float, default=[0.9,0.999], nargs='+')
        self.parser.add_argument('--eps', type=float, default=1e-08)
        self.parser.add_argument('--amsgrad', action='store_true')

        # SGD
        self.parser.add_argument('--momentum', type=float, default=0.9)

        # Model
        self.parser.add_argument('--inputsz', type=int, default=None, nargs='+')
        self.parser.add_argument('--outputsz', type=int, default=None, nargs='+')
        self.parser.add_argument('--fov', type=vec3, default=(20,256,256))
        self.parser.add_argument('--depth', type=int, default=4)
        self.parser.add_argument('--long_range', action='store_true')
        self.parser.add_argument('--symmetric', action='store_true')
        self.parser.add_argument('--group', type=int, default=0)

        # Data augmentation
        self.parser.add_argument('--recompute', action='store_true')
        self.parser.add_argument('--grayscale', action='store_true')
        self.parser.add_argument('--warping', action='store_true')
        self.parser.add_argument('--misalign', action='store_true')
        self.parser.add_argument('--missing', type=int, default=0)
        self.parser.add_argument('--blur', type=int, default=0)
        self.parser.add_argument('--box', default=None)
        self.parser.add_argument('--lost', action='store_true')
        self.parser.add_argument('--random_fill', action='store_true')
        self.parser.add_argument('--skip_track', type=float, default=0.0)

        # Multiclass detection
        self.parser.add_argument('--aff', type=float, default=0)
        self.parser.add_argument('--psd', type=float, default=0)
        self.parser.add_argument('--mit', type=float, default=0)

        # Metric learning
        self.parser.add_argument('--vec', type=float, default=0)
        self.parser.add_argument('--embed_dim', type=int, default=10)

        # Onnx export
        self.parser.add_argument('--onnx', action='store_true')

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

        # Optimizer
        if opt.optim == 'Adam':
            optim_keys = ['lr','betas','eps','amsgrad']
        elif opt.optim == 'SGD':
            optim_keys = ['lr','momentum']
        else:
            optim_keys = ['lr']
        args = vars(opt)
        opt.optim_params = {k: v for k, v in args.items() if k in optim_keys}

        # Loss
        opt.loss_params = dict()
        opt.loss_params['size_average'] = opt.size_average
        opt.loss_params['margin0'] = opt.margin0
        opt.loss_params['margin1'] = opt.margin1
        opt.loss_params['inverse'] = opt.inverse
        opt.loss_params['logits'] = not opt.no_logits

        # Model
        opt.fov = tuple(opt.fov)
        #defaults -> copy fov
        opt.inputsz = opt.fov if opt.inputsz is None else tuple(opt.inputsz)
        opt.outputsz = opt.fov if opt.outputsz is None else tuple(opt.outputsz)
        opt.in_spec = dict(input=(1,) + opt.inputsz)
        opt.edges = self.get_edges(opt)
        opt.out_spec = dict()
        opt.loss_weight = dict()

        if opt.vec > 0:
            opt.out_spec['embedding'] = (opt.embed_dim,) + opt.outputsz
            opt.loss_weight['embedding'] = opt.vec
        else:
            if opt.aff > 0:
                opt.out_spec['affinity'] = (len(opt.edges),) + opt.outputsz
                opt.loss_weight['affinity'] = opt.aff

            if opt.psd > 0:
                opt.out_spec['synapse'] = (1,) + opt.outputsz
                opt.loss_weight['synapse'] = opt.psd

            if opt.mit > 0:
                opt.out_spec['mitochondria'] = (1,) + opt.outputsz
                opt.loss_weight['mitochondria'] = opt.mit

        assert len(opt.out_spec) > 0
        assert len(opt.out_spec) == len(opt.loss_weight)

        # Data augmentation
        opt.aug_params = dict()
        opt.aug_params['recompute'] = opt.recompute
        opt.aug_params['grayscale'] = opt.grayscale
        opt.aug_params['warping'] = opt.warping
        opt.aug_params['misalign'] = opt.misalign
        opt.aug_params['missing'] = opt.missing
        opt.aug_params['blur'] = opt.blur
        opt.aug_params['box'] = opt.box
        opt.aug_params['lost'] = opt.lost
        opt.aug_params['random'] = opt.random_fill
        opt.aug_params['skip_track'] = opt.skip_track

        # Multiclass detection
        opt.data_params = dict()
        opt.data_params['seg'] = opt.aff > 0 or opt.vec > 0
        opt.data_params['psd'] = opt.psd > 0
        opt.data_params['mit'] = opt.mit > 0
        opt.data_params['pad_size'] = opt.pad_size
        assert(len(opt.pad_size) == 3 and all(x >= 0 for x in opt.pad_size))

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
