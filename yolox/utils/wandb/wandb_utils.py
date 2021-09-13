#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This file contains the utilities for logging on Weights & Biases
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from tqdm import tqdm

import cv2

import torch

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES

import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Union
from utils import get_rank

__all__ = ["WandBLogger"]


FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[3].as_posix())  # add yolox/ to path


class WandBLogger:
    """
    The `WandBLogger` provides an easy integration with
    Weights & Biases logging. Each monitored metric is automatically
    logged to a dedicated Weights & Biases project dashboard.

    .. note::
    The wandb log files are placed by default in "./wandb/" unless specified.
    """

    def __init__(
        self,
        project_name: str = "YOLOX",
        run_name: str = None,
        save_code: bool = True,
        config: object = None,
        dir: Union[str, Path] = None,
        model: object = None,
        params: dict = None,
    ) -> None:
        """
        Creates an instance of the `WandBLogger`.
        :param project_name: Name of the W&B project.
        :param run_name: Name of the W&B run.
        :param save_code: Saves the main training script to W&B.
        :param dir: Path to the local log directory for W&B logs to be saved at.
        :param config: Syncs hyper-parameters and config values used to W&B.
        :param params: All arguments for wandb.init() function call.
        Visit https://docs.wandb.ai/ref/python/init to learn about all
        wand.init() parameters.
        """

        self.project_name = project_name
        self.run_name = run_name
        self.save_code = save_code
        self.dir = dir
        self.config = config
        self.model = model
        self.params = params

        self._import_wandb()
        self._args_parse()
        self._before_job()

    def _import_wandb(self):
        try:
            import wandb

            assert hasattr(wandb, "__version__")
        except (ImportError, AssertionError):
            raise ImportError('Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def _args_parse(self):
        self.init_kwargs = {
            "project": self.project_name,
            "name": self.run_name,
            "save_code": self.save_code,
            "dir": self.dir,
            "config": vars(self.config),
        }
        if self.params:
            self.init_kwargs.update(self.params)

    def _before_job(self):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()
        if self.model is not None:
            self.wandb.watch(self.model)
        self.wandb.run._label(repo=self.project_name)

    def log_metrics(self, log_dict: dict = None) -> None:
        for key, value in log_dict.items():

            curr_val = value[-1]

            if isinstance(curr_val, (int, float, Tensor)):
                self.wandb.log({key: curr_val})

            else:
                return
    
    def log_dataset() -> None:
        pass

    def log_checkpoint() -> None:
        pass

    def get_dataset() -> object:
        pass

    def resume_train() -> object:
        pass

    def log_video() -> None:
        pass

    def log_pred(self, image, bbox, class_id) -> None:
        # load raw input photo
        raw_image = load_img(filename, target_size=(log_height, log_width))
        all_boxes = []
        # plot each bounding box for this image
        for b_i, box in enumerate(v_boxes):
            # get coordinates and labels
            box_data = {"position" : {
            "minX" : box.xmin,
            "maxX" : box.xmax,
            "minY" : box.ymin,
            "maxY" : box.ymax},
            "class_id" : display_ids[v_labels[b_i]],
            # optionally caption each box with its class and score
            "box_caption" : "%s (%.3f)" % (v_labels[b_i], v_scores[b_i]),
            "domain" : "pixel",
            "scores" : { "score" : v_scores[b_i] }}
            all_boxes.append(box_data)

        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = self.wandb.Image(raw_image, boxes = {"predictions": {"box_data": all_boxes, "class_labels" :class_id_to_label}})
        return box_image




def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = F.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue
        # Get score and class with highest confidence
        class_conf = F.max(image_pred[:, 5 : 5 + num_classes], 1, keepdims=True)
        class_pred = F.argmax(image_pred[:, 5 : 5 + num_classes], 1, keepdims=True)

        class_conf_squeeze = F.squeeze(class_conf)
        conf_mask = image_pred[:, 4] * class_conf_squeeze >= conf_thre
        detections = F.concat((image_pred[:, :5], class_conf, class_pred), 1)
        detections = detections[conf_mask]
        if not detections.shape[0]:
            continue

        nms_out_index = F.vision.nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = F.concat((output[i], detections))

    return output



class Predictor(object):
    def __init__(
        self,
        model,
        confthre=0.01,
        nmsthre=0.65,
        test_size=(640, 640),
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
    ): 
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = 80
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.test_size = test_size



    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
            if img is None:
                raise ValueError("test image path is invalid!")
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preprocess(img, self.test_size)
        img_info["ratio"] = ratio
        img = F.expand_dims(torch.tensor(img), 0)

        outputs = self.model(img)
        outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.numpy()

        # preprocessing: resize
        bboxes = output[:, 0:4] / ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res
