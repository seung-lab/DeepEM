from __future__ import print_function
import imp
import os
import time

import torch
from tensorboardX import SummaryWriter

from deepem.train.option import TrainOptions
from deepem.train.utils import load_model, load_data
from deepem.utils.monitor import LearningMonitor


def train(opt):
    # Model
    model = load_model(opt)

    # Data loaders
    train_loader, val_loader = load_data(opt)

    # Optimizer
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable, lr=opt.base_lr)

    # Create a summary writer.
    # writer = SummaryWriter(opt.log_dir)

    # Training loop
    t0 = time.time()
    print("========== BEGIN TRAINING LOOP ==========")
    for i in range(opt.chkpt_num, opt.max_iter):

        # Load training samples.
        sample = train_loader()

        # Optimizer step
        optimizer.zero_grad()
        losses, nmasks, preds = model(sample)
        losses = {k: v.mean() for k, v in losses.items()}
        nmasks = {k: v.mean() for k, v in nmasks.items()}
        loss = sum([w*losses[k] for k, w in opt.loss_weight.items()])
        loss.backward()
        optimizer.step()

        # Elapsed time
        elapsed = time.time() - t0

        # Dispaly
        disp = "Iter: %8d, " % (i+1)
        for k, v in losses.items():
            disp += "%s = %.3f, " % (k, v.item())
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
