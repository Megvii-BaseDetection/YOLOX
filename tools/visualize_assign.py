#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import sys
import random
import time
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.exp import Exp, get_exp
from yolox.core import Trainer
from yolox.utils import configure_module, configure_omp
from yolox.tools.train import make_parser


class AssignVisualizer(Trainer):

    def __init__(self, exp: Exp, args):
        super().__init__(exp, args)
        self.batch_cnt = 0
        self.vis_dir = os.path.join(self.file_name, "vis")
        os.makedirs(self.vis_dir, exist_ok=True)

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            path_prefix = os.path.join(self.vis_dir, f"assign_vis_{self.batch_cnt}_")
            self.model.visualize(inps, targets, path_prefix)

        if self.use_model_ema:
            self.ema_model.update(self.model)

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
        )
        self.batch_cnt += 1
        if self.batch_cnt >= self.args.max_batch:
            sys.exit(0)

    def after_train(self):
        logger.info("Finish visualize assignment, exit...")


def assign_vis_parser():
    parser = make_parser()
    parser.add_argument("--max-batch", type=int, default=1, help="max batch of images to visualize")
    return parser


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_omp()
    cudnn.benchmark = True

    visualizer = AssignVisualizer(exp, args)
    visualizer.train()


if __name__ == "__main__":
    configure_module()
    args = assign_vis_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    main(exp, args)
