from __future__ import print_function
import imp
import os
import time

import torch

from tensorboardX import SummaryWriter

from pytorch_connectome.dataiter import DataIter
from pytorch_connectome.options import TrainOptions
from pytorch_connectome.utils.monitor import LearningMonitor


def train(opt):
    # TODO: Create a model.

    # Data
    data = load_data(opt)
    train_data = DataIter(opt, data, is_train=True)
    # eval_data = DataIter(opt, data, is_train=False)

    # Optimizer
    # trainable = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.Adam(trainable, lr=opt.base_lr)

    # Create a summary writer.
    # writer = SummaryWriter(opt.log_dir)

    # Training loop
    t0 = time.time()
    print("========== BEGIN TRAINING LOOP ==========")
    for i in range(opt.chkpt_num, opt.max_iter):

        # Load training samples.
        sample = train_data(opt.in_spec)

        # # Optimizer step
        # optimizer.zero_grad()
        # # losses, nmasks, inputs, preds, labels = model(sample)
        # weights = [opt.loss_weight[k] for k in sorted(opt.loss_weight)]
        # loss = sum([w*l.mean() for w, l in zip(weights,losses)])
        # loss.backward()
        # optimizer.step()

        # Elapsed time
        elapsed = time.time() - t0

        # Dispaly
        disp = "Iter: %8d, " % (i+1)
        disp += "lr = %.6f, " % opt.base_lr
        disp += "(elapsed = %.3f). " % elapsed
        print(disp)

        # Averaging & displaying stats
        if (i+1) % opt.avgs_intv == 0 or i < opt.warm_up:
            pass

        # Logging images
        if (i+1) % opt.imgs_intv == 0:
            pass

        # Evaluation loop
        if (i+1) % opt.eval_intv == 0:
            pass

        # Model snapshot
        if (i+1) % opt.chkpt_intv == 0:
            pass

        # Restart timer.
        t0 = time.time()

    # Close the summary writer.
    # writer.close()


def load_data(opt):
    # Train & validation data IDs
    mod = imp.load_source('sampler', opt.sampler)
    train_ids = mod.get_data_ids(True)
    val_ids = mod.get_data_ids(False)
    data_ids = list(set().union(train_ids, val_ids))

    # Load data.
    mod = imp.load_source('data', opt.data)
    return mod.load_data(opt.data_dir, data_ids=data_ids)


if __name__ == "__main__":
    # Options
    opt = TrainOptions().parse()

    # GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

    # Make directories.
    if not os.path.isdir(opt.exp_dir):
        os.makedirs(opt.exp_dir)
    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)
    if not os.path.isdir(opt.model_dir):
        os.makedirs(opt.model_dir)

    # cuDNN auto-tuning
    torch.backends.cudnn.benchmark = opt.autotune

    # Run experiment.
    print("Running experiment: {}".format(opt.exp_name))
    train(opt)
