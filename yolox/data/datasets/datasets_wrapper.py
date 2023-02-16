#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import bisect
import copy
import os
import random
from abc import ABCMeta, abstractmethod
from functools import partial, wraps
from multiprocessing.pool import ThreadPool
import psutil
from loguru import logger
from tqdm import tqdm

import numpy as np

from torch.utils.data.dataset import ConcatDataset as torchConcatDataset
from torch.utils.data.dataset import Dataset as torchDataset


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
    def mosaic_getitem(getitem_fn):
        """
        Decorator method that needs to be used around the ``__getitem__`` method. |br|
        This decorator enables the closing mosaic

        Example:
            >>> class CustomSet(ln.data.Dataset):
            ...     def __len__(self):
            ...         return 10
            ...     @ln.data.Dataset.mosaic_getitem
            ...     def __getitem__(self, index):
            ...         return self.enable_mosaic
        """

        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                self.enable_mosaic = index[0]
                index = index[1]

            ret_val = getitem_fn(self, index)

            return ret_val

        return wrapper


class CacheDataset(Dataset, metaclass=ABCMeta):
    """ This class is a subclass of the base :class:`yolox.data.datasets.Dataset`,
    that enables cache images to ram or disk.

    Args:
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
        num_imgs (int): datset size
        data_dir (str): the root directory of the dataset, e.g. `/path/to/COCO`.
        cache_dir_name (str): the name of the directory to cache to disk,
            e.g. `"custom_cache"`. The files cached to disk will be saved
            under `/path/to/COCO/custom_cache`.
        path_filename (str): a list of paths to the data relative to the `data_dir`,
            e.g. if you have data `/path/to/COCO/train/1.jpg`, `/path/to/COCO/train/2.jpg`,
            then `path_filename = ['train/1.jpg', ' train/2.jpg']`.
        cache (bool): whether to cache the images to ram or disk.
        cache_type (str): the type of cache,
            "ram" : Caching imgs to ram for fast training.
            "disk": Caching imgs to disk for fast training.
    """

    def __init__(
        self,
        input_dimension,
        num_imgs=None,
        data_dir=None,
        cache_dir_name=None,
        path_filename=None,
        cache=False,
        cache_type="ram",
    ):
        super().__init__(input_dimension)
        self.cache = cache
        self.cache_type = cache_type

        if self.cache and self.cache_type == "disk":
            self.cache_dir = os.path.join(data_dir, cache_dir_name)
            self.path_filename = path_filename

        if self.cache and self.cache_type == "ram":
            self.imgs = None

        if self.cache:
            self.cache_images(
                num_imgs=num_imgs,
                data_dir=data_dir,
                cache_dir_name=cache_dir_name,
                path_filename=path_filename,
            )

    def __del__(self):
        if self.cache and self.cache_type == "ram":
            del self.imgs

    @abstractmethod
    def read_img(self, index):
        """
        Given index, return the corresponding image

        Args:
            index (int): image index
        """
        raise NotImplementedError

    def cache_images(
        self,
        num_imgs=None,
        data_dir=None,
        cache_dir_name=None,
        path_filename=None,
    ):
        assert num_imgs is not None, "num_imgs must be specified as the size of the dataset"
        if self.cache_type == "disk":
            assert (data_dir and cache_dir_name and path_filename) is not None, \
                "data_dir, cache_name and path_filename must be specified if cache_type is disk"
            self.path_filename = path_filename

        mem = psutil.virtual_memory()
        mem_required = self.cal_cache_occupy(num_imgs)
        gb = 1 << 30

        if self.cache_type == "ram":
            if mem_required > mem.available:
                self.cache = False
            else:
                logger.info(
                    f"{mem_required / gb:.1f}GB RAM required, "
                    f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB RAM available, "
                    f"Since the first thing we do is cache, "
                    f"there is no guarantee that the remaining memory space is sufficient"
                )

        if self.cache and self.imgs is None:
            if self.cache_type == 'ram':
                self.imgs = [None] * num_imgs
                logger.info("You are using cached images in RAM to accelerate training!")
            else:   # 'disk'
                if not os.path.exists(self.cache_dir):
                    os.mkdir(self.cache_dir)
                    logger.warning(
                        f"\n*******************************************************************\n"
                        f"You are using cached images in DISK to accelerate training.\n"
                        f"This requires large DISK space.\n"
                        f"Make sure you have {mem_required / gb:.1f} "
                        f"available DISK space for training your dataset.\n"
                        f"*******************************************************************\\n"
                    )
                else:
                    logger.info(f"Found disk cache at {self.cache_dir}")
                    return

            logger.info(
                "Caching images...\n"
                "This might take some time for your dataset"
            )

            num_threads = min(8, max(1, os.cpu_count() - 1))
            b = 0
            load_imgs = ThreadPool(num_threads).imap(
                partial(self.read_img, use_cache=False),
                range(num_imgs)
            )
            pbar = tqdm(enumerate(load_imgs), total=num_imgs)
            for i, x in pbar:   # x = self.read_img(self, i, use_cache=False)
                if self.cache_type == 'ram':
                    self.imgs[i] = x
                else:   # 'disk'
                    cache_filename = f'{self.path_filename[i].split(".")[0]}.npy'
                    cache_path_filename = os.path.join(self.cache_dir, cache_filename)
                    os.makedirs(os.path.dirname(cache_path_filename), exist_ok=True)
                    np.save(cache_path_filename, x)
                b += x.nbytes
                pbar.desc = \
                    f'Caching images ({b / gb:.1f}/{mem_required / gb:.1f}GB {self.cache_type})'
            pbar.close()

    def cal_cache_occupy(self, num_imgs):
        cache_bytes = 0
        num_samples = min(num_imgs, 32)
        for _ in range(num_samples):
            img = self.read_img(index=random.randint(0, num_imgs - 1), use_cache=False)
            cache_bytes += img.nbytes
        mem_required = cache_bytes * num_imgs / num_samples
        return mem_required


def cache_read_img(use_cache=True):
    def decorator(read_img_fn):
        """
        Decorate the read_img function to cache the image

        Args:
            read_img_fn: read_img function
            use_cache (bool, optional): For the decorated read_img function,
                whether to read the image from cache.
                Defaults to True.
        """
        @wraps(read_img_fn)
        def wrapper(self, index, use_cache=use_cache):
            cache = self.cache and use_cache
            if cache:
                if self.cache_type == "ram":
                    img = self.imgs[index]
                    img = copy.deepcopy(img)
                elif self.cache_type == "disk":
                    img = np.load(
                        os.path.join(
                            self.cache_dir, f"{self.path_filename[index].split('.')[0]}.npy"))
                else:
                    raise ValueError(f"Unknown cache type: {self.cache_type}")
            else:
                img = read_img_fn(self, index)
            return img
        return wrapper
    return decorator
