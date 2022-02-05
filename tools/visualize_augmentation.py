#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time

import matplotlib.pyplot as plt
import torchvision
from loguru import logger

from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Augmentation Visualizer")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        required=True,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument(
        "-c",
        "--cls_file",
        default=None,
        required=True,
        type=str,
        help="pls input your list of classes file separated by newline",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        required=True,
        type=str,
        help="pls input where to save augmented images",
    )
    parser.add_argument("-n", "--samples", default=None, type=int, help="Samples to visualize")
    # parser.add_argument("-b", "--draw-bbox", action='store_true', help="Draw bbox on images")
    return parser


@logger.catch
def main(exp, args):
    loader = exp.get_data_loader(
        batch_size=1,
        is_distributed=False,
        no_aug=False,
        cache_img=False,
    )

    with open(args.cls_file, 'r', encoding='utf-8') as f_in:
        classes = f_in.read().split()

    for i, (img, target, *rest) in enumerate(loader, 1):
        # Remove batch dimension and cast it to uint8
        img = img.squeeze().byte()
        if args.draw_bbox:
            target = target.squeeze()
            target = target[~target.eq(0).all(dim=1)]
            labels = [classes[int(bbox[0])] for bbox in target.tolist()]
            img = torchvision.utils.draw_bounding_boxes(img, target[:, 1:], labels)
        # Change CHW -> HWC and BGR -> RGB
        img = img.permute(1, 2, 0)[..., (2, 1, 0)]
        plt.imshow(img)
        plt.savefig(os.path.join(args.output_dir, time.strftime("%Y%m%d-%H%M%S.png")))
        if args.samples and args.samples == i:
            break


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    main(exp, args)
