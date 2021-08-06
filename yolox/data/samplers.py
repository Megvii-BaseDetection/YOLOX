#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import itertools
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler


class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (dim, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will prepend a dimension, whilst ensuring it stays the same across one mini-batch.
    """

    def __init__(self, *args, input_dimension=None, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dimension
        self.new_input_dim = None
        self.mosaic = mosaic

    def __iter__(self):
        self.__set_input_dim()
        for batch in super().__iter__():
            yield [(self.input_dim, idx, self.mosaic) for idx in batch]
            self.__set_input_dim()

    def __set_input_dim(self):
        """ This function randomly changes the the input dimension of the dataset. """
        if self.new_input_dim is not None:
            self.input_dim = (self.new_input_dim[0], self.new_input_dim[1])
            self.new_input_dim = None


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size
