#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
from collections import OrderedDict

import megengine as mge
import torch


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, help="path of weight file")
    parser.add_argument(
        "-o",
        "--output",
        default="weight_mge.pkl",
        type=str,
        help="path of weight file",
    )
    return parser


def numpy_weights(weight_file):
    torch_weights = torch.load(weight_file, map_location="cpu")
    if "model" in torch_weights:
        torch_weights = torch_weights["model"]
    new_dict = OrderedDict()
    for k, v in torch_weights.items():
        new_dict[k] = v.cpu().numpy()
    return new_dict


def map_weights(weight_file, output_file):
    torch_weights = numpy_weights(weight_file)

    new_dict = OrderedDict()
    for k, v in torch_weights.items():
        if "num_batches_tracked" in k:
            print("drop: {}".format(k))
            continue
        if k.endswith("bias"):
            print("bias key: {}".format(k))
            v = v.reshape(1, -1, 1, 1)
            new_dict[k] = v
        elif "dconv" in k and "conv.weight" in k:
            print("depthwise conv key: {}".format(k))
            cout, cin, k1, k2 = v.shape
            v = v.reshape(cout, 1, cin, k1, k2)
            new_dict[k] = v
        else:
            new_dict[k] = v

    mge.save(new_dict, output_file)
    print("save weights to {}".format(output_file))


def main():
    parser = make_parser()
    args = parser.parse_args()
    map_weights(args.weights, args.output)


if __name__ == "__main__":
    main()
