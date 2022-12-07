#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO

import copy
import os
import psutil
import random

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        self.imgs = None
        self.cache = cache

        self._cache_images()

    def _cache_images(self):
        # cache == 'ram' & check failed -> cache False
        # else keep cache unchanged
        if self.cache == "ram" and not self.check_cache_ram():
            self.cache = False
        if self.cache and self.imgs is None:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            if self.cache == 'ram':
                self.imgs = [None] * self.num_imgs
                logger.info("You are using cached images in RAM to accelerate training!")
            else:   # 'disk'
                self.cache_dir = os.path.join(
                    self.data_dir,
                    f"{self.name}_cache{self.img_size[0]}x{self.img_size[1]}"
                )
                if not os.path.exists(self.cache_dir):
                    os.mkdir(self.cache_dir)
                    logger.warning(
                        "\n*******************************************************************\n"
                        "You are using cached images in DISK to accelerate training.\n"
                        "This requires large DISK space.\n"
                        "Make sure you have 136G available DISK space for training COCO.\n"
                        "*******************************************************************\n"
                    )
                else:
                    logger.info("Found disk cache!")
                    return

            logger.info("Caching images for the first time. \
                        This might take about 15 minutes for COCO")

            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool
            # number of YOLOX multiprocessing threads
            NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
            TQDM_BAR_FORMAT = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}'  # tqdm bar format
            results = ThreadPool(NUM_THREADS).imap(self.load_resized_img, range(self.num_imgs))
            pbar = tqdm(enumerate(results), total=self.num_imgs, bar_format=TQDM_BAR_FORMAT)
            for i, x in pbar:   # x = self.load_resized_img(self, i)
                if self.cache == 'ram':
                    self.imgs[i] = x
                else:   # 'disk'
                    cache_filename = f'{self.annotations[i]["filename"].split(".")[0]}.npy'
                    np.save(os.path.join(self.cache_dir, cache_filename), x)
                b += x.nbytes
                pbar.desc = f'Caching images ({b / gb:.1f}GB {self.cache})'
            pbar.close()

    def __len__(self):
        return self.num_imgs

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def check_cache_ram(self, safety_margin=0.1):
        # Check image caching requirements vs available memory
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.num_imgs, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = self.load_resized_img(random.randint(0, self.num_imgs - 1))  # sample image
            b += im.nbytes
        mem_required = b * self.num_imgs / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available
        logger.info(
            f"{mem_required / gb:.1f}GB RAM required, "
            f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, "
            f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
        )
        return cache

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return {
            'label': res,
            'resized_image_info': resized_info,
            'original_image_size': img_info,
            'filename': file_name
        }

    def load_anno(self, index):
        return self.annotations[index]['label']

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
        file_name = self.annotations[index]['filename']
        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    def pull_item(self, index):
        id_ = self.ids[index]
        label = self.annotations[index]['label']
        origin_image_size = self.annotations[index]['original_image_size']
        filename = self.annotations[index]['filename']

        if self.cache == 'ram':
            img = self.imgs[index]
        elif self.cache == 'disk':
            img = np.load(os.path.join(self.cache_dir, f"{filename.split('.')[0]}.npy"))
        else:
            img = self.load_resized_img(index)

        return copy.deepcopy(img), copy.deepcopy(label), origin_image_size, np.array([id_])

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
