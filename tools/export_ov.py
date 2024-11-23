#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import openvino as ov

def make_parser():
    parser = argparse.ArgumentParser("YOLOX OpenVINO deploy")
    parser.add_argument(
        "-o", "--output_name", type=str, default="yolox.xml", help="output name of models"
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    
    if os.path.isfile(args.name):
        ov_model = ov.convert_model(args.name)
        ov.save_model(ov_model, args.output_name)
        logger.info("generated openvino model named {}".format(args.output_name))
    else:
        logger.error("Please run export_onnx.py to export to ONNX model first")


if __name__ == "__main__":
    main()
