#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from torch.utils.data.dataset import ConcatDataset as torchConcatDataset
from torch.utils.data.dataset import Dataset as torchDataset

import bisect
from functools import wraps


class ConcatDataset(torchConcatDataset):
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        if hasattr(self.datasets[0], "input_dim"):
            self._input_dim = self.datasets[0].input_dim
            self.input_dim = self.datasets[0].input_dim

    def pull_item(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].pull_item(sample_idx)


class MixConcatDataset(torchConcatDataset):
    def __init__(self, datasets):
        super(MixConcatDataset, self).__init__(datasets)
        if hasattr(self.datasets[0], "input_dim"):
            self._input_dim = self.datasets[0].input_dim
            self.input_dim = self.datasets[0].input_dim

    def __getitem__(self, index):

        if not isinstance(index, int):
            idx = index[1]
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        if not isinstance(index, int):
            index = (index[0], sample_idx, index[2])

        return self.datasets[dataset_idx][index]


class Dataset(torchDataset):
    """ This class is a subclass of the base :class:`torch.utils.data.Dataset`,
    that enables on the fly resizing of the ``input_dim``.

    Args:
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
    """

    def __init__(self, input_dimension, mosaic=True):
        super().__init__()
        self.__input_dim = input_dimension[:2]
        self.enable_mosaic = mosaic

    @property
    def input_dim(self):
        """
        Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth
        for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim
        return self.__input_dim

    @staticmethod
    def resize_getitem(getitem_fn):
        """
        Decorator method that needs to be used around the ``__getitem__`` method. |br|
        This decorator enables the on the fly resizing of
        the ``input_dim`` with our :class:`~lightnet.data.DataLoader` class.

        Example:
            >>> class CustomSet(ln.data.Dataset):
            ...     def __len__(self):
            ...         return 10
            ...     @ln.data.Dataset.resize_getitem
            ...     def __getitem__(self, index):
            ...         # Should return (image, anno) but here we return input_dim
            ...         return self.input_dim
            >>> data = CustomSet((200,200))
            >>> data[0]
            (200, 200)
            >>> data[(480,320), 0]
            (480, 320)
        """

        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                has_dim = True
                self._input_dim = index[0]
                self.enable_mosaic = index[2]
                index = index[1]
            else:
                has_dim = False

            ret_val = getitem_fn(self, index)

            if has_dim:
                del self._input_dim

            return ret_val

        return wrapper
