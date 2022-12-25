#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This file comes from
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/fast_eval_api.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Megvii Inc. All rights reserved.

import copy
import time

import numpy as np
from pycocotools.cocoeval import COCOeval

from .jit_ops import FastCOCOEvalOp


class COCOeval_opt(COCOeval):
    """
    This is a slightly modified version of the original COCO API, where the functions evaluateImg()
    and accumulate() are implemented in C++ to speedup evaluation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = FastCOCOEvalOp().load()

    def evaluate(self):
        """
        Run per image evaluation on given images and store results in self.evalImgs_cpp, a
        datastructure that isn't readable from Python but is used by a c++ implementation of
        accumulate().  Unlike the original COCO PythonAPI, we don't populate the datastructure
        self.evalImgs because this datastructure is a computational bottleneck.
        :return: None
        """
        tic = time.time()

        print("Running per image evaluation...")
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print(
                "useSegm (deprecated) is not None. Running {} evaluation".format(
                    p.iouType
                )
            )
        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()

        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            computeIoU = self.computeIoU
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks
        self.ious = {
            (imgId, catId): computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        maxDet = p.maxDets[-1]

        # <<<< Beginning of code differences with original COCO API
        def convert_instances_to_cpp(instances, is_det=False):
            # Convert annotations for a list of instances in an image to a format that's fast
            # to access in C++
            instances_cpp = []
            for instance in instances:
                instance_cpp = self.module.InstanceAnnotation(
                    int(instance["id"]),
                    instance["score"] if is_det else instance.get("score", 0.0),
                    instance["area"],
                    bool(instance.get("iscrowd", 0)),
                    bool(instance.get("ignore", 0)),
                )
                instances_cpp.append(instance_cpp)
            return instances_cpp

        # Convert GT annotations, detections, and IOUs to a format that's fast to access in C++
        ground_truth_instances = [
            [convert_instances_to_cpp(self._gts[imgId, catId]) for catId in p.catIds]
            for imgId in p.imgIds
        ]
        detected_instances = [
            [
                convert_instances_to_cpp(self._dts[imgId, catId], is_det=True)
                for catId in p.catIds
            ]
            for imgId in p.imgIds
        ]
        ious = [[self.ious[imgId, catId] for catId in catIds] for imgId in p.imgIds]

        if not p.useCats:
            # For each image, flatten per-category lists into a single list
            ground_truth_instances = [
                [[o for c in i for o in c]] for i in ground_truth_instances
            ]
            detected_instances = [
                [[o for c in i for o in c]] for i in detected_instances
            ]

        # Call C++ implementation of self.evaluateImgs()
        self._evalImgs_cpp = self.module.COCOevalEvaluateImages(
            p.areaRng,
            maxDet,
            p.iouThrs,
            ious,
            ground_truth_instances,
            detected_instances,
        )
        self._evalImgs = None

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("COCOeval_opt.evaluate() finished in {:0.2f} seconds.".format(toc - tic))
        # >>>> End of code differences with original COCO API

    def accumulate(self):
        """
        Accumulate per image evaluation results and store the result in self.eval.  Does not
        support changing parameter settings from those used by self.evaluate()
        """
        print("Accumulating evaluation results...")
        tic = time.time()
        if not hasattr(self, "_evalImgs_cpp"):
            print("Please run evaluate() first")

        self.eval = self.module.COCOevalAccumulate(self._paramsEval, self._evalImgs_cpp)

        # recall is num_iou_thresholds X num_categories X num_area_ranges X num_max_detections
        self.eval["recall"] = np.array(self.eval["recall"]).reshape(
            self.eval["counts"][:1] + self.eval["counts"][2:]
        )

        # precision and scores are num_iou_thresholds X num_recall_thresholds X num_categories X
        # num_area_ranges X num_max_detections
        self.eval["precision"] = np.array(self.eval["precision"]).reshape(
            self.eval["counts"]
        )
        self.eval["scores"] = np.array(self.eval["scores"]).reshape(self.eval["counts"])
        toc = time.time()
        print(
            "COCOeval_opt.accumulate() finished in {:0.2f} seconds.".format(toc - tic)
        )
