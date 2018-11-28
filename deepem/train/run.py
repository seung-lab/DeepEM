from __future__ import print_function
import imp
import os
import time

import torch

from deepem.train.logger import Logger
from deepem.train.option import Options
from deepem.train.utils import *


def train(opt):
    # Model
    model = load_model(opt)

    # Optimizer
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = load_optimizer(opt, trainable)

    # Data loaders
    train_loader, val_loader = load_data(opt)

    # Initial checkpoint
    save_chkpt(model, opt.model_dir, opt.chkpt_num, optimizer)

    # Training loop
    print("========== BEGIN TRAINING LOOP ==========")
    with Logger(opt) as logger:

        # Timer
        t0 = time.time()

        for i in range(opt.chkpt_num, opt.max_iter):

            # Load training samples.
            sample = train_loader()

            # Optimizer step
            optimizer.zero_grad()
            losses, nmasks, preds = forward(model, sample, opt)
            total_loss = sum([w*losses[k] for k, w in opt.loss_weight.items()])
            total_loss.backward()
            optimizer.step()

            # Elapsed time
            elapsed = time.time() - t0

            # Record keeping
            logger.record('train', losses, nmasks, elapsed=elapsed)

            # Log & display averaged stats.
            if (i+1) % opt.avgs_intv == 0 or i < opt.warm_up:
                logger.check('train', i+1)

            # Image logging
            if (i+1) % opt.imgs_intv == 0:
                logger.log_images('train', i+1, preds, sample)

            # Evaluation loop
            if (i+1) % opt.eval_intv == 0:
                eval_loop(i+1, model, val_loader, opt, logger)

            # Model checkpoint
            if (i+1) % opt.chkpt_intv == 0:
                save_chkpt(model, opt.model_dir, i+1, optimizer)

            # Reset timer.
            t0 = time.time()


def eval_loop(iter_num, model, data_loader, opt, logger):
    if not opt.no_eval:
        model.eval()

    # Evaluation loop
    print("---------- BEGIN EVALUATION LOOP ----------")
    with torch.no_grad():
        t0 = time.time()
        for i in range(opt.eval_iter):
            sample = data_loader()
            losses, nmasks, preds = forward(model, sample, opt)
            elapsed = time.time() - t0

            # Record keeping
            logger.record('test', losses, nmasks, elapsed=elapsed)

            # Restart timer.
            t0 = time.time()

    # Log & display averaged stats.
    logger.check('test', iter_num)
    print("-------------------------------------------")

    model.train()


if __name__ == "__main__":

    # Options
    opt = Options().parse()

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
    torch.backends.cudnn.benchmark = not opt.no_autotune

    # Run experiment.
    print("Running experiment: {}".format(opt.exp_name))
    train(opt)
