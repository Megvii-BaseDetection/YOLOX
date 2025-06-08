#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.utils.dist import barrier, deinit_distributed, init_distributed


__all__ = ["launch"]


def launch(
    main_func,
    args=()
):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        args (tuple): arguments passed to main_func
    """
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    if world_size > 1:
        init_distributed(world_size=world_size, rank=rank)

        barrier()
        main_func(*args)
        barrier()
        
        deinit_distributed()
    else:
        main_func(*args)