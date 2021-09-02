#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This file contains the utilities for logging on Weights & Biases
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from tqdm import tqdm

import logging
import os
import sys
import yaml
from contextlib import contextmanager
from pathlib import Path
from utils import get_rank

__all__ = ["WandBLogger"]


FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[3].as_posix())  # add yolox/ to path


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """source - https://gist.github.com/simon-weber/7853144
    A context manager that will prevent any logging messages triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL is defined.
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


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
