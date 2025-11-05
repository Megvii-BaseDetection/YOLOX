#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from xml.parsers.expat import model
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    

def make_parser():
    parser = argparse.ArgumentParser("YOLOX executorch deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.pte", help="output name of models"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    exported_program = torch.export.export(model, (dummy_input,))

    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()]  # CPU | CoreMLPartitioner() for iOS | QnnPartitioner() for Qualcomm
    ).to_executorch()

    with open(args.output_name, "wb") as f:
        f.write(program.buffer)

    logger.info("generated executorch model named {}".format(args.output_name))

if __name__ == "__main__":
    main()
