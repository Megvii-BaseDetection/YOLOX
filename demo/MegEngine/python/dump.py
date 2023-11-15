#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse

import megengine as mge
import numpy as np
from megengine import jit

from build import build_and_load


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo Dump")
    parser.add_argument("-n", "--name", type=str, default="yolox-s", help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--dump_path", default="model.mge", help="path to save the dumped model"
    )
    return parser


def dump_static_graph(model, graph_name="model.mge"):
    model.eval()
    model.head.decode_in_inference = False

    data = mge.Tensor(np.random.random((1, 3, 640, 640)))

    @jit.trace(capture_as_const=True)
    def pred_func(data):
        outputs = model(data)
        return outputs

    pred_func(data)
    pred_func.dump(
        graph_name,
        arg_names=["data"],
        optimize_for_inference=True,
        enable_fuse_conv_bias_nonlinearity=True,
    )


def main(args):
    model = build_and_load(args.ckpt, name=args.name)
    dump_static_graph(model, args.dump_path)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
