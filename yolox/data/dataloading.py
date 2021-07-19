#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random

import torch
from torch.utils.data.dataloader import DataLoader as torchDataLoader
from torch.utils.data.dataloader import default_collate

from .samplers import YoloBatchSampler


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


class DataLoader(torchDataLoader):
    """
    Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py

    Note:
        This dataloader only works with :class:`lightnet.data.Dataset` based datasets.

    Example:
        >>> class CustomSet(ln.data.Dataset):
        ...     def __len__(self):
        ...         return 4
        ...     @ln.data.Dataset.resize_getitem
        ...     def __getitem__(self, index):
        ...         # Should return (image, anno) but here we return (input_dim,)
        ...         return (self.input_dim,)
        >>> dl = ln.data.DataLoader(
        ...     CustomSet((200,200)),
        ...     batch_size = 2,
        ...     collate_fn = ln.data.list_collate   # We want the data to be grouped as a list
        ... )
        >>> dl.dataset.input_dim    # Default input_dim
        (200, 200)
        >>> for d in dl:
        ...     d
        [[(200, 200), (200, 200)]]
        [[(200, 200), (200, 200)]]
        >>> dl.change_input_dim(320, random_range=None)
        (320, 320)
        >>> for d in dl:
        ...     d
        [[(320, 320), (320, 320)]]
        [[(320, 320), (320, 320)]]
        >>> dl.change_input_dim((480, 320), random_range=None)
        (480, 320)
        >>> for d in dl:
        ...     d
        [[(480, 320), (480, 320)]]
        [[(480, 320), (480, 320)]]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if len(args) > 5:
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        elif len(args) > 3:
            shuffle = args[2]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        else:
            if "shuffle" in kwargs:
                shuffle = kwargs["shuffle"]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                    # sampler = torch.utils.data.DistributedSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = YoloBatchSampler(
                sampler,
                self.batch_size,
                self.drop_last,
                input_dimension=self.dataset.input_dim,
            )
            # batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations =

        self.batch_sampler = batch_sampler

        self.__initialized = True

    def close_mosaic(self):
        self.batch_sampler.mosaic = False

    def change_input_dim(self, multiple=32, random_range=(10, 19)):
        """ This function will compute a new size and update it on the next mini_batch.

        Args:
            multiple (int or tuple, optional): values to multiply the randomly generated range by.
                Default **32**
            random_range (tuple, optional): This (min, max) tuple sets the range
                for the randomisation; Default **(10, 19)**

        Return:
            tuple: width, height tuple with new dimension

        Note:
            The new size is generated as follows: |br|
            First we compute a random integer inside ``[random_range]``.
            We then multiply that number with the ``multiple`` argument,
            which gives our final new input size. |br|
            If ``multiple`` is an integer we generate a square size. If you give a tuple
            of **(width, height)**, the size is computed
            as :math:`rng * multiple[0], rng * multiple[1]`.

        Note:
            You can set the ``random_range`` argument to **None** to set
            an exact size of multiply. |br|
            See the example above for how this works.
        """
        if random_range is None:
            size = 1
        else:
            size = random.randint(*random_range)

        if isinstance(multiple, int):
            size = (size * multiple, size * multiple)
        else:
            size = (size * multiple[0], size * multiple[1])

        self.batch_sampler.new_input_dim = size

        return size


def list_collate(batch):
    """
    Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader, if you want to have a list of
    items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    items = list(zip(*batch))

    for i in range(len(items)):
        if isinstance(items[i][0], (list, tuple)):
            items[i] = list(items[i])
        else:
            items[i] = default_collate(items[i])

    return items
