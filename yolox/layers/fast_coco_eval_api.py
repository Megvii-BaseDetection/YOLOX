#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This file comes from
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/fast_eval_api.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Megvii Inc. All rights reserved.

import copy
import time

import numpy as np
from pycocotools.cocoeval import COCOeval

from .jit_ops import FastCOCOEvalOp


class COCOeval_opt(COCOeval):
    """
    This is a slightly modified version of the original COCO API, where the functions evaluateImg()
    and accumulate() are implemented in C++ to speedup evaluation
    """
    pass
