#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random
import uuid

import numpy as np

import torch

__all__ = [
    "close_mosaic",
    "get_yolox_datadir",
    "worker_init_reset_seed",
]


def close_mosaic(dataloader):
    assert hasattr(dataloader.dataset, "enable_mosaic"), "mosaic setting not found"
    dataloader.dataset.enable_mosaic = False
    return iter(dataloader)  # to refresh dataloader


def get_yolox_datadir():
    """
    get dataset dir of YOLOX. If environment variable named `YOLOX_DATADIR` is set,
    this function will return value of the environment variable. Otherwise, use data
    """
    yolox_datadir = os.getenv("YOLOX_DATADIR", None)
    if yolox_datadir is None:
        import yolox
        yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
        yolox_datadir = os.path.join(yolox_path, "datasets")
    return yolox_datadir


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)
