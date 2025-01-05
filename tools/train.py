#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from pathlib import Path
import yaml

import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices


class TrainConfigs:
    def __init__(self, yaml_args):
        self.experiment_name = yaml_args.get("experiment_name", None)
        self.name = yaml_args.get("name", self.experiment_name)
        self.exp_file = yaml_args.get("exp_file", None)
        self.batch_size = yaml_args.get("batch_size", 16)
        self.devices = yaml_args.get("devices", None)
        self.resume = yaml_args.get("resume", False)
        self.ckpt = yaml_args.get("ckpt", None)
        self.start_epoch = yaml_args.get("start_epoch", None)
        self.fp16 = yaml_args.get("fp16", False)
        self.num_machines = yaml_args.get("num_machines", 1)
        self.machine_rank = yaml_args.get("machine_rank", 0)
        self.cache = yaml_args.get("cache", "ram")
        self.occupy = yaml_args.get("occupy", False)
        self.logger = yaml_args.get("logger", "tensorboard")
        self.opts = yaml_args.get("opts", [])
        self.dist_backend = yaml_args.get("dist_backend", "nccl")
        self.dist_url = yaml_args.get("dist_url", None)


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
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()

    configure_module()

    with open(script_dir / "train.yaml", "r") as f:
        args = yaml.safe_load(f)
    args = TrainConfigs(args)
    
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    
    check_exp_value(exp)

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
