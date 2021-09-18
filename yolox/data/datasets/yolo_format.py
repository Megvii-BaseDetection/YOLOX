#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger
from tqdm import tqdm

import cv2
import numpy as np

from .datasets_wrapper import Dataset


class YoloFormatDataset(Dataset):
    """
    Dataset with data and annotations like YOLO.
    """

    def __init__(
        self, data_dir, anno_file="train.txt",
        normalized_anno=True, anno_format="cxcywh",
        label_first_anno=True, img_size=(416, 416),
        preproc=None, cache=False,
    ):
        """
        YOLO dataformat dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): path of dataset folder.
            anno_file (str): path of annotations. If not given, use `labels` under data_dir folder.
            normalized_anno (bool): annotation data is normalized or not. Default to `True`.
            anno_format (string): format of annotation. Now support
                `xyxy`: (x1, y1, x2, y2)
                `cxcywh`: (centerX, centerY, width, height).
                Default to "cxcywh".
            label_first_anno (bool): label is the 1st of annotation or not. Default to `True`.
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy.
            cache (bool): cache images or not.
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.anno_file = anno_file
        self.normalized_anno = normalized_anno
        assert anno_format in ("cxcywh", "xyxy"), "unspported format {}".format(anno_format)
        self.anno_format = anno_format
        self.label_first_anno = label_first_anno

        self.image_path = self.generate_image_path()
        # TODO add class label file and backgroud label
        self.name = "YoloFormat"
        self.img_size = img_size
        self.annotations = self._load_annotations()
        self.preproc = preproc
        self.cached_imgs = None
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.image_path)

    def __del__(self):
        del self.cached_imgs

    def generate_image_path(self):
        with open(os.path.join(self.data_dir, self.anno_file), "r") as f:
            path_list = [_.strip() for _ in f.readlines()]
        return [os.path.join(self.data_dir, p) for p in path_list]

    def _load_annotations(self):
        logger.info("loading annoations...")
        # TODO: multiprocess loading could be faster
        return [self.load_anno_from_img_path(path) for path in tqdm(self.image_path)]

    def load_anno_from_img_path(self, path):
        """
        Args:
            path(str): path of image.
        """
        def get_anno_filename(path, suffix="txt"):
            path_list = []
            for x in path.split(os.sep):
                if x == "images":
                    x = "labels"
                path_list.append(x)
            # change suffix
            path_list[-1] = ".".join(path_list[-1].split(".")[:-1] + [suffix])
            return os.sep.join(path_list)

        anno_file = get_anno_filename(path)
        # suppose that only contains 4 coord and 1 label
        anno = np.loadtxt(anno_file).reshape(-1, 5)
        if self.label_first_anno:
            # label to last dim
            anno = np.concatenate((anno[..., 1:], anno[..., None, 0]), axis=-1)

        img = cv2.imread(path)
        img_name = os.path.basename(path)
        assert img is not None, "Could not find image named {}".format(path)
        height, width, _ = img.shape

        if self.normalized_anno:
            anno[..., 0:4:2] *= width
            anno[..., 1:4:2] *= height

        # convert different anno format to xyxy
        if self.anno_format == "cxcywh":
            anno[..., 0] -= anno[..., 2] / 2
            anno[..., 1] -= anno[..., 3] / 2

        scale = min(self.img_size[0] / height, self.img_size[1] / width)
        anno[..., 0:4] *= scale

        img_info = (height, width)
        resized_info = (int(height * scale), int(width * scale))

        return (anno, img_info, resized_info, img_name)

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have enough RAM and disk space to train your own dataset.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.cached_imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.image_path[index]
        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, "Could not found image named: {}".format(file_name)

        return img

    def pull_item(self, index):
        res, img_info, resized_info, _ = self.annotations[index]
        if self.cached_imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([index])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
