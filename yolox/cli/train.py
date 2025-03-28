# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import sys
import warnings

import torch
from loguru import logger
from torch.backends import cudnn

from yolox.config import YoloxConfig
from yolox.core import launch
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

from .utils import parse_model_config_opts, resolve_config


def make_parser():
    parser = argparse.ArgumentParser("yolox train")
    parser.add_argument("-c", "--config", type=str, help="A builtin config such as yolox_s, or a custom Python class given as {module}:{classname} such as yolox.config:YoloxS")
    parser.add_argument("-n", "--name", type=str, default=None, help="Model name; defaults to the model name specified in config")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
                Implemented loggers include `tensorboard`, `mlflow` and `wandb`.",
        default="tensorboard"
    )
    parser.add_argument(
        "-D",
        type=str,
        metavar="OPT=VALUE",
        help="Override model configuration option with custom value (example: -D num_classes=71)",
        action="append",
    )
    return parser


def train(config: YoloxConfig, args):
    if config.seed is not None:
        assert isinstance(config.seed, int)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
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

    trainer = config.get_trainer(args)
    trainer.train()


def main(argv: list[str]) -> None:
    configure_module()
    args = make_parser().parse_args(argv)
    if args.config is None:
        raise AttributeError("Please specify a model configuration.")
    config = resolve_config(args.config)
    config.update(parse_model_config_opts(args.D))
    config.validate()

    if not args.name:
        args.name = config.name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        config.dataset = config.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        train,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(config, args),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
