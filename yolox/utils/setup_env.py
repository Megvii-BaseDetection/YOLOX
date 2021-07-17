#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import resource
import subprocess

import cv2

__all__ = ["configure_nccl", "configure_module"]


def configure_nccl():
    """Configure multi-machine environment variables of NCCL."""
    os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
    os.environ["GLOO_SOCKET_IFNAME"] = "ib0"
    os.environ["NCCL_IB_DISABLE"] = "1"

    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "cd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; > /dev/null"
    )
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"


def configure_module(ulimit_value=8192):
    """
    Configure pytorch module environment. setting of ulimit and cv2 will be set.

    Args:
        ulimit_value(int): default open file number on linux. Default value: 4096.
    """
    # system setting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, rlimit[1]))
    # cv2
    # multiprocess might be harmful on performance of torch dataloader
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
