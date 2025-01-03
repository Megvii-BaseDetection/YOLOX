#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This file mainly comes from
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/comm.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Megvii Inc. All rights reserved.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import os
import pickle
import time
from contextlib import contextmanager
from loguru import logger

import numpy as np

import torch
from torch import distributed as dist
from yolox.utils.device_utils import get_current_device, get_distributed_backend, get_distributed_init_method, get_local_device_count, get_xla_model, xm

__all__ = [
    "wait_for_the_master",
    "is_main_process",
    "synchronize",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "get_local_size",
    "time_synchronized",
    "gather",
    "all_gather",
]

__DEFAULT_GLOO_GROUP = None
xm = get_xla_model()

@contextmanager
def wait_for_the_master(local_rank: int = None):
    """
    Make all processes waiting for the master to do some task.

    Args:
        local_rank (int): the rank of the current process. Default to None.
            If None, it will use the rank of the current process.
    """
    if local_rank is None:
        local_rank = get_local_rank()

    if local_rank > 0:
        barrier()
    yield
    if local_rank == 0:
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        else:
            barrier()


def synchronize(group=None):
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return
    
    barrier(group=group)


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local machine
    """
    return int(os.getenv("LOCAL_RANK", 0))


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group, i.e. the number of processes per machine.
    """
    return get_local_device_count()


def is_main_process() -> bool:
    return get_rank() == 0

def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    global __DEFAULT_GLOO_GROUP
    assert __DEFAULT_GLOO_GROUP is not None, "Gloo group is not initialized"
    return __DEFAULT_GLOO_GROUP
    
def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    device = torch.device("cpu") if backend == "gloo" else get_current_device()

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)
    synchronize(group=group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    synchronize(group=group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if dist.get_world_size(group=group) == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    synchronize(group=group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    synchronize(group=group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def barrier(group=None):
    dist.barrier(group=group)

def init_distributed(world_size: int, rank: int):

    if not dist.is_initialized():
        init_method = get_distributed_init_method()
        backend = get_distributed_backend()  

        dist.init_process_group(backend=backend, 
                                    world_size=world_size, 
                                    rank=rank, 
                                    init_method=init_method)
        
        global __DEFAULT_GLOO_GROUP
        if __DEFAULT_GLOO_GROUP is None:
            __DEFAULT_GLOO_GROUP = dist.new_group(backend="gloo")

def deinit_distributed():
    if dist.is_initialized():
        global __DEFAULT_GLOO_GROUP
        try:
            if __DEFAULT_GLOO_GROUP is not None:
                dist.destroy_process_group(group=__DEFAULT_GLOO_GROUP)
        except Exception as e:
            logger.warning(f"Error: {e}")
        finally:
            dist.destroy_process_group()
       