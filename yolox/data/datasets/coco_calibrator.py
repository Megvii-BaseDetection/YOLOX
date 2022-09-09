#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset
import torch
import random
from glob import glob

class COCOCalibratorDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        num_images=200,
        seed=10
    ):
        super().__init__(img_size)
        path = os.path.join(data_dir, name)
        self.files = glob(path + "/*.jpg")
        random.seed(seed)
        random.shuffle(self.files)
        self.files = self.files[:num_images]
        self.nf=len(self.files)
    
        self.img_size = img_size
        self.preproc = preproc
        
        assert self.nf > 0
        
    def __len__(self):
        return self.nf
    
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
        img = cv2.imread(self.files[index])

        assert img is not None, f"file named {self.files[index]} not found"

        return img

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img = self.load_resized_img(index)

        if self.preproc is not None:
            img, _ = self.preproc(img, None, self.input_dim)
        return [torch.Tensor(img)]
