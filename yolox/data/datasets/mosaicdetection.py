#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random

import cv2
import numpy as np

from yolox.utils import adjust_box_anns

from ..data_augment import box_candidates, random_perspective
from .datasets_wrapper import Dataset


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset.

    Parameters
    ----------
    dataset : Pytorch Dataset
        Gluon dataset object.
    *args : list
        Additional arguments for mixup random sampler.
    """

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True, *args
    ):
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self._mosaic = mosaic
        self.enable_mixup = enable_mixup

    def __len__(self):
        return len(self._dataset)

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        if self._mosaic:
            labels4 = []
            s = self._dataset.input_dim[0]
            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * s, 1.5 * s))
            xc = int(random.uniform(0.5 * s, 1.5 * s))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i, index in enumerate(indices):
                img, _labels, _, _ = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                r = 1.0 * s / max(h0, w0)  # resize image to img_size
                interp = cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
                (h, w) = img.shape[:2]

                if i == 0:  # top left
                    # base image with 4 tiles
                    img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                    # xmin, ymin, xmax, ymax (large image)
                    x1a, y1a, x2a, y2a = (max(xc - w, 0), max(yc - h, 0), xc, yc,)
                    # xmin, ymin, xmax, ymax (small image)
                    x1b, y1b, x2b, y2b = (w - (x2a - x1a), h - (y2a - y1a), w, h,)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
                padw = x1a - x1b
                padh = y1a - y1b

                labels = _labels.copy()  # [[xmin, ymin, xmax, ymax, label_ind], ... ]
                if _labels.size > 0:  # Normalized xywh to pixel xyxy format
                    labels[:, 0] = r * _labels[:, 0] + padw
                    labels[:, 1] = r * _labels[:, 1] + padh
                    labels[:, 2] = r * _labels[:, 2] + padw
                    labels[:, 3] = r * _labels[:, 3] + padh
                labels4.append(labels)

            if len(labels4):
                labels4 = np.concatenate(labels4, 0)
                np.clip(labels4[:, :4], 0, 2 * s, out=labels4[:, :4])  # use with random_affine
            img4, labels4 = random_perspective(
                img4,
                labels4,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-s // 2, -s // 2],
            )  # border to remove

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if self.enable_mixup and not len(labels4) == 0:
                img4, labels4 = self.mixup(img4, labels4, self.input_dim)
            mix_img, padded_labels = self.preproc(img4, labels4, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            return mix_img, padded_labels, img_info, int(idx)

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, idx = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, int(idx)

    def mixup(self, origin_img, origin_labels, input_dim):
        # jit_factor = random.uniform(0.8, 1.2)
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            id_ = self._dataset.ids[cp_index]
            anno_ids = self._dataset.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
            cp_labels = self._dataset.coco.loadAnns(anno_ids)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
        else:
            cp_img = np.ones(input_dim) * 114.0
        cp_scale_ratio = input_dim[0] / max(img.shape[0], img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4], cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5]
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
