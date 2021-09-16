#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This file contains the utilities for logging on Weights & Biases
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from tqdm import tqdm

import cv2

import torch

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import get_rank

import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Union

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
        self.log_dict = None

        self._import_wandb()
        self._args_parse()
        self._before_job()
        
        self.current_epoch = 0
        self.result_table = self.wandb.Table(["epoch", "prediction", "avg_confidence"])

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
            "config": vars(self.config) if self.config else None,
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
        self.log_dict = log_dict
        for key, value in log_dict.items():

            if isinstance(value, (int, float, Tensor)):
                self.wandb.log({key: value})

            else:
                return

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

        class_set = self.wandb.Classes([{'id': id, 'name': name} for id, name in class_labels.items()])

        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = self.wandb.Image(
            image,
            boxes={
                "predictions": {
                    "box_data": all_boxes,
                    "class_labels": class_labels,
                },
            },
            classes=class_set
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

        self.wandb.log({"Result Table": self.result_table})

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

    def check_and_upload_dataset(self, opt):
        """
        Check if the dataset format is compatible and upload it as W&B artifact
        arguments:
        opt (namespace)-- Commandline arguments for current run
        returns:
        Updated dataset info dictionary where local dataset paths are replaced by WAND_ARFACT_PREFIX links.
        """
        assert self.wandb, "Install wandb to upload dataset"
        config_path = self.log_dataset_artifact(
            opt.data,
            opt.single_cls,
            "YOLOX" if opt.project == "runs/train" else Path(opt.project).stem,
        )
        print("Created dataset config file ", config_path)
        with open(config_path, errors="ignore") as f:
            wandb_data_dict = yaml.safe_load(f)
        return wandb_data_dict

    def log_dataset_artifact(
        self, data_file, single_cls, project, overwrite_config=False
    ):
        """
        Log the dataset as W&B artifact and return the new data file with W&B links
        arguments:
        data_file (str) -- the .yaml file with information about the dataset like - path, classes etc.
        single_class (boolean)  -- train multi-class data as single-class
        project (str) -- project name. Used to construct the artifact path
        overwrite_config (boolean) -- overwrites the data.yaml file if set to true otherwise creates a new
        file with _wandb postfix. Eg -> data_wandb.yaml
        returns:
        the new .yaml file with artifact links. it can be used to start training directly from artifacts
        """
        self.data_dict = check_dataset(data_file)  # parse and check
        data = dict(self.data_dict)
        nc, names = (1, ["item"]) if single_cls else (int(data["nc"]), data["names"])
        names = {k: v for k, v in enumerate(names)}  # to index dictionary
        self.train_artifact = (
            self.create_dataset_table(
                LoadImagesAndLabels(data["train"], rect=True, batch_size=1),
                names,
                name="train",
            )
            if data.get("train")
            else None
        )
        self.val_artifact = (
            self.create_dataset_table(
                LoadImagesAndLabels(data["val"], rect=True, batch_size=1),
                names,
                name="val",
            )
            if data.get("val")
            else None
        )
        if data.get("train"):
            data["train"] = WANDB_ARTIFACT_PREFIX + str(Path(project) / "train")
        if data.get("val"):
            data["val"] = WANDB_ARTIFACT_PREFIX + str(Path(project) / "val")
        path = Path(data_file).stem
        path = (
            path if overwrite_config else path + "_wandb"
        ) + ".yaml"  # updated data.yaml path
        data.pop("download", None)
        data.pop("path", None)
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

        if self.job_type == "Training":  # builds correct artifact pipeline graph
            self.wandb_run.use_artifact(self.val_artifact)
            self.wandb_run.use_artifact(self.train_artifact)
            self.val_artifact.wait()
            self.val_table = self.val_artifact.get("val")
            self.map_val_table_path()
        else:
            self.wandb_run.log_artifact(self.train_artifact)
            self.wandb_run.log_artifact(self.val_artifact)
        return path

    def log_checkpoint() -> None:
        pass

    def get_dataset() -> object:
        pass

    def resume_train() -> object:
        pass
