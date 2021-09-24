#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This file contains the utilities for logging on Weights & Biases
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from tqdm import tqdm

import cv2

import torch
from torch import Tensor

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES

import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Union

__all__ = ["WandBLogger"]

WANDB_ARTIFACT_PREFIX = "wandb-artifact://"

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
        job_type: str = "training",
        params: dict = None,
    ) -> None:
        """
        Creates an instance of the `WandBLogger`.
        :param project_name: Name of the W&B project.
        :param run_name: Name of the W&B run.
        :param save_code: Saves the main training script to W&B.
        :param dir: Path to the local log directory for W&B logs to be saved at.
        :param model: Model checkpoint to be logged to W&B.
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
        self.job_type = job_type
        self.params = params

        self._import_wandb()
        self._args_parse()
        self._before_job()

        self.current_epoch = 0
        self.result_table = self.wandb.Table(["epoch", "prediction", "avg_confidence"])

    def _import_wandb(self):
        """Imports Weights & Biases package

        Raises:
            ImportError: If the Weights & Biases package is not installed.
        """
        try:
            import wandb

            assert hasattr(wandb, "__version__")
        except (ImportError, AssertionError):
            raise ImportError('Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def _args_parse(self):
        """Parses the arguments for wandb.init() function call."""
        self.init_kwargs = {
            "project": self.project_name,
            "name": self.run_name,
            "save_code": self.save_code,
            "dir": self.dir,
            "job_type": self.job_type,
            "config": vars(self.config) if self.config else None,
        }
        if self.params:
            self.init_kwargs.update(self.params)

    def _before_job(self):
        """Initializes the Weights & Biases logger."""
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()
        if self.model is not None:
            self.wandb.watch(self.model)
        self.wandb.run._label(repo="YOLOX")

    def log_metrics(self, key: str = None, value: Union[int, float, Tensor] = None, step: Union[int, float] = None) -> None:
        """Logs metrics to Weights & Biases dashboard.

        Args:
            key: Name of the metric.
            value: Value of the metric.
        """
        self.wandb.log({key: value}, step = step)

    def _handle_pred(self, image, output):
        """Log a single prediction."""

        bboxes = output[:, 0:4]
        labels = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        # load input image
        if isinstance(image, str):
            filename = os.path.basename(image)
            image = cv2.imread(image)
            if image is None:
                raise ValueError("test image path is invalid!")
        else:
            filename = None

        all_boxes = []

        # plot each bounding box for this image
        for b_i, box in enumerate(bboxes):

            # get coordinates and labels
            box_data = {
                "position": {
                    "minX": int(box[0]),
                    "maxX": int(box[2]),
                    "minY": int(box[1]),
                    "maxY": int(box[3]),
                },
                "class_id": int(labels[b_i]),
                # optionally caption each box with its class and score
                # "box_caption": "%s (%.3f)" % (v_labels[b_i], v_scores[b_i]),
                "domain": "pixel",
                "scores": {"score": float(scores[b_i])},
            }
            all_boxes.append(box_data)

        ids = [i for i in range(len(COCO_CLASSES))]
        class_labels = dict(zip(ids, list(COCO_CLASSES)))

        class_set = self.wandb.Classes(
            [{"id": id, "name": name} for id, name in class_labels.items()]
        )

        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = self.wandb.Image(
            image,
            boxes={
                "predictions": {
                    "box_data": all_boxes,
                    "class_labels": class_labels,
                },
            },
            classes=class_set,
        )

        return box_image

    def log_pred(self, image, output) -> None:
        """Log a prediction on a single image."""

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("image must be a torch.Tensor or a numpy.ndarray")

        pred = self._handle_pred(image, output)
        scores = output[:, 4] * output[:, 5]
        avg_score = scores.mean()

        self.result_table.add_data(self.current_epoch, pred, avg_score)

    def log_preds(self, images, outputs) -> None:
        """Log a batch of predictions."""

        predictions = []

        if isinstance(images, torch.Tensor):
            if len(images.shape) == 4:
                for image, output in zip(images, outputs):
                    if output is not None:
                        self.log_pred(image, output)
            else:
                raise ValueError("images must be a torch.Tensor of shape (N, C, H, W)")

        self.wandb.log({"Prediction Table": self.result_table})

    def create_dataset_artifact(self, opt) -> None:
        """Create a dataset artifact for the current run."""

        assert self.wandb, "Install wandb to upload dataset"

        nc, names = (1, ["item"]) if opt.single_cls else len(COCO_CLASSES), COCO_CLASSES

        self.train_artifact = self.create_dataset_table(opt.train, nc, names)
        self.val_artifact = self.create_dataset_table(opt.val, nc, names)

        self.wandb.log({"Train Dataset": self.train_artifact})
        self.wandb.log({"Val Dataset": self.val_artifact})

    def create_dataset_table(self, path, names, name="") -> wandb.Table:
        """Create a wandb table for the dataset."""

        artifact = self.wandb.Artifact(name=name, type="dataset")

        # img_files = tqdm([path]) if isinstance(dataset.path, str) and Path(dataset.path).is_dir() else None
        # img_files = tqdm(dataset.img_files) if not img_files else img_files

        for img_file in img_files:
            if Path(img_file).is_dir():
                artifact.add_dir(img_file, name="data/images")
                labels_path = "labels".join(dataset.path.rsplit("images", 1))
                artifact.add_dir(labels_path, name="data/labels")
            else:
                artifact.add_file(img_file, name="data/images/" + Path(img_file).name)
                label_file = Path(img2label_paths([img_file])[0])
                artifact.add_file(
                    str(label_file), name="data/labels/" + label_file.name
                ) if label_file.exists() else None

        table = self.wandb.Table(columns=["id", "train_image", "Classes", "name"])
        class_set = self.wandb.Classes(
            [{"id": id, "name": name} for id, name in class_to_id.items()]
        )

        for si, (img, labels, paths, shapes) in enumerate(tqdm(dataset)):
            box_data, img_classes = [], {}
            for cls, *xywh in labels[:, 1:].tolist():
                cls = int(cls)
                box_data.append(
                    {
                        "position": {
                            "middle": [xywh[0], xywh[1]],
                            "width": xywh[2],
                            "height": xywh[3],
                        },
                        "class_id": cls,
                        "box_caption": "%s" % (class_to_id[cls]),
                    }
                )
                img_classes[cls] = class_to_id[cls]
            boxes = {
                "ground_truth": {"box_data": box_data, "class_labels": class_to_id}
            }  # inference-space
            table.add_data(
                si,
                self.wandb.Image(paths, classes=class_set, boxes=boxes),
                list(img_classes.values()),
                Path(paths).name,
            )
        artifact.add(table, name)
        return artifact

    def log_checkpoint(self, path, best_model=False):
        """
        Log the model checkpoint as W&B artifact
        arguments:
        path (Path)   -- Path of directory containing the checkpoints
        opt (namespace) -- Command line arguments for this run
        epoch (int)  -- Current epoch number
        fitness_score (float) -- fitness score for current epoch
        best_model (boolean) -- Boolean representing if the current checkpoint is the best yet.
        """
        path = Path(path)
        model_artifact = self.wandb.Artifact(
            "run_" + self.wandb.run.id + "_model",
            type="model",
            metadata={
                "original_url": str(path),
                # 'epochs_trained': epoch + 1,
                # 'save period': opt.save_period,
                # 'project': opt.project,
                # 'total_epochs': opt.epochs,
                # 'fitness_score': fitness_score
            },
        )
        model_artifact.add_file(str(path), name="last.pth")
        self.wandb.log_artifact(
            model_artifact,
            aliases=[
                "latest",
                "last",
                "epoch " + str(self.current_epoch),
                "best" if best_model else "",
            ],
        )
        # print("Saving model artifact on epoch ", epoch + 1)

    def resume_train() -> object:
        pass
