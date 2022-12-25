#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import inspect
import os
import sys
from collections import defaultdict
from loguru import logger

import cv2
import numpy as np

import torch


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass

    def isatty(self):
        # when using colab, jax is installed by default and issue like
        # https://github.com/Megvii-BaseDetection/YOLOX/issues/1437 might be raised
        # due to missing attribute like`isatty`.
        # For more details, checked the following link:
        # https://github.com/google/jax/blob/10720258ea7fb5bde997dfa2f3f71135ab7a6733/jax/_src/pretty_printer.py#L54  # noqa
        return True


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")


class WandbLogger(object):
    """
    Log training runs, datasets, models, and predictions to Weights & Biases.
    This logger sends information to W&B at wandb.ai.
    By default, this information includes hyperparameters,
    system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    For more information, please refer to:
    https://docs.wandb.ai/guides/track
    https://docs.wandb.ai/guides/integrations/other/yolox
    """
    def __init__(self,
                 project=None,
                 name=None,
                 id=None,
                 entity=None,
                 save_dir=None,
                 config=None,
                 val_dataset=None,
                 num_eval_images=100,
                 log_checkpoints=False,
                 **kwargs):
        """
        Args:
            project (str): wandb project name.
            name (str): wandb run name.
            id (str): wandb run id.
            entity (str): wandb entity name.
            save_dir (str): save directory.
            config (dict): config dict.
            val_dataset (Dataset): validation dataset.
            num_eval_images (int): number of images from the validation set to log.
            log_checkpoints (bool): log checkpoints
            **kwargs: other kwargs.

        Usage:
            Any arguments for wandb.init can be provided on the command line using
            the prefix `wandb-`.
            Example
            ```
            python tools/train.py .... --logger wandb wandb-project <project-name> \
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_imges <num-images> \
                wandb-log_checkpoints <bool>
            ```
            The val_dataset argument is not open to the command line.
        """
        try:
            import wandb
            self.wandb = wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "wandb is not installed."
                "Please install wandb using pip install wandb"
                )

        from yolox.data.datasets import VOCDetection

        self.project = project
        self.name = name
        self.id = id
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self.entity = entity
        self._run = None
        self.val_artifact = None
        if num_eval_images == -1:
            self.num_log_images = len(val_dataset)
        else:
            self.num_log_images = min(num_eval_images, len(val_dataset))
        self.log_checkpoints = (log_checkpoints == "True" or log_checkpoints == "true")
        self._wandb_init = dict(
            project=self.project,
            name=self.name,
            id=self.id,
            entity=self.entity,
            dir=self.save_dir,
            resume="allow"
        )
        self._wandb_init.update(**kwargs)

        _ = self.run

        if self.config:
            self.run.config.update(self.config)
        self.run.define_metric("train/epoch")
        self.run.define_metric("val/*", step_metric="train/epoch")
        self.run.define_metric("train/step")
        self.run.define_metric("train/*", step_metric="train/step")

        self.voc_dataset = VOCDetection

        if val_dataset and self.num_log_images != 0:
            self.val_dataset = val_dataset
            self.cats = val_dataset.cats
            self.id_to_class = {
                cls['id']: cls['name'] for cls in self.cats
            }
            self._log_validation_set(val_dataset)

    @property
    def run(self):
        if self._run is None:
            if self.wandb.run is not None:
                logger.info(
                    "There is a wandb run already in progress "
                    "and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()`"
                    "before instantiating `WandbLogger`."
                )
                self._run = self.wandb.run
            else:
                self._run = self.wandb.init(**self._wandb_init)
        return self._run

    def _log_validation_set(self, val_dataset):
        """
        Log validation set to wandb.

        Args:
            val_dataset (Dataset): validation dataset.
        """
        if self.val_artifact is None:
            self.val_artifact = self.wandb.Artifact(name="validation_images", type="dataset")
            self.val_table = self.wandb.Table(columns=["id", "input"])

            for i in range(self.num_log_images):
                data_point = val_dataset[i]
                img = data_point[0]
                id = data_point[3]
                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if isinstance(id, torch.Tensor):
                    id = id.item()

                self.val_table.add_data(
                    id,
                    self.wandb.Image(img)
                )

            self.val_artifact.add(self.val_table, "validation_images_table")
            self.run.use_artifact(self.val_artifact)
            self.val_artifact.wait()

    def _convert_prediction_format(self, predictions):
        image_wise_data = defaultdict(int)

        for key, val in predictions.items():
            img_id = key

            try:
                bboxes, cls, scores = val
            except KeyError:
                bboxes, cls, scores = val["bboxes"], val["categories"], val["scores"]

            # These store information of actual bounding boxes i.e. the ones which are not None
            act_box = []
            act_scores = []
            act_cls = []

            if bboxes is not None:
                for box, classes, score in zip(bboxes, cls, scores):
                    if box is None or score is None or classes is None:
                        continue
                    act_box.append(box)
                    act_scores.append(score)
                    act_cls.append(classes)

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in act_box],
                    "scores": [score.numpy().item() for score in act_scores],
                    "categories": [
                        self.val_dataset.class_ids[int(act_cls[ind])]
                        for ind in range(len(act_box))
                    ],
                }
            })

        return image_wise_data

    def log_metrics(self, metrics, step=None):
        """
        Args:
            metrics (dict): metrics dict.
            step (int): step number.
        """

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.item()

        if step is not None:
            metrics.update({"train/step": step})
            self.run.log(metrics)
        else:
            self.run.log(metrics)

    def log_images(self, predictions):
        if len(predictions) == 0 or self.val_artifact is None or self.num_log_images == 0:
            return

        table_ref = self.val_artifact.get("validation_images_table")

        columns = ["id", "predicted"]
        for cls in self.cats:
            columns.append(cls["name"])

        if isinstance(self.val_dataset, self.voc_dataset):
            predictions = self._convert_prediction_format(predictions)

        result_table = self.wandb.Table(columns=columns)

        for idx, val in table_ref.iterrows():

            avg_scores = defaultdict(int)
            num_occurrences = defaultdict(int)

            id = val[0]
            if isinstance(id, list):
                id = id[0]

            if id in predictions:
                prediction = predictions[id]
                boxes = []
                for i in range(len(prediction["bboxes"])):
                    bbox = prediction["bboxes"][i]
                    x0 = bbox[0]
                    y0 = bbox[1]
                    x1 = bbox[2]
                    y1 = bbox[3]
                    box = {
                        "position": {
                            "minX": min(x0, x1),
                            "minY": min(y0, y1),
                            "maxX": max(x0, x1),
                            "maxY": max(y0, y1)
                        },
                        "class_id": prediction["categories"][i],
                        "domain": "pixel"
                    }
                    avg_scores[
                        self.id_to_class[prediction["categories"][i]]
                    ] += prediction["scores"][i]
                    num_occurrences[self.id_to_class[prediction["categories"][i]]] += 1
                    boxes.append(box)
            else:
                boxes = []
            average_class_score = []
            for cls in self.cats:
                if cls["name"] not in num_occurrences:
                    score = 0
                else:
                    score = avg_scores[cls["name"]] / num_occurrences[cls["name"]]
                average_class_score.append(score)
            result_table.add_data(
                idx,
                self.wandb.Image(val[1], boxes={
                        "prediction": {
                            "box_data": boxes,
                            "class_labels": self.id_to_class
                        }
                    }
                ),
                *average_class_score
            )

        self.wandb.log({"val_results/result_table": result_table})

    def save_checkpoint(self, save_dir, model_name, is_best, metadata=None):
        """
        Args:
            save_dir (str): save directory.
            model_name (str): model name.
            is_best (bool): whether the model is the best model.
            metadata (dict): metadata to save corresponding to the checkpoint.
        """

        if not self.log_checkpoints:
            return

        if "epoch" in metadata:
            epoch = metadata["epoch"]
        else:
            epoch = None

        filename = os.path.join(save_dir, model_name + "_ckpt.pth")
        artifact = self.wandb.Artifact(
            name=f"run_{self.run.id}_model",
            type="model",
            metadata=metadata
        )
        artifact.add_file(filename, name="model_ckpt.pth")

        aliases = ["latest"]

        if is_best:
            aliases.append("best")

        if epoch:
            aliases.append(f"epoch-{epoch}")

        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self):
        self.run.finish()

    @classmethod
    def initialize_wandb_logger(cls, args, exp, val_dataset):
        wandb_params = dict()
        prefix = "wandb-"
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            if k.startswith("wandb-"):
                try:
                    wandb_params.update({k[len(prefix):]: int(v)})
                except ValueError:
                    wandb_params.update({k[len(prefix):]: v})

        return cls(config=vars(exp), val_dataset=val_dataset, **wandb_params)
