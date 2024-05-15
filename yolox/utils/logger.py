#!/usr/bin/env python3
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
        # flush is related with CPR(cursor position report) in terminal
        return sys.__stdout__.flush()

    def isatty(self):
        # when using colab, jax is installed by default and issue like
        # https://github.com/Megvii-BaseDetection/YOLOX/issues/1437 might be raised
        # due to missing attribute like`isatty`.
        # For more details, checked the following link:
        # https://github.com/google/jax/blob/10720258ea7fb5bde997dfa2f3f71135ab7a6733/jax/_src/pretty_printer.py#L54  # noqa
        return sys.__stdout__.isatty()

    def fileno(self):
        # To solve the issue when using debug tools like pdb
        return sys.__stdout__.fileno()


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

import importlib.metadata
import importlib.util
from collections.abc import MutableMapping
import packaging.version
import json, platform, time

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}



class MlflowLogger(object):
    """
    Logging training runs with hyperparamter, datasets and trained models to MLflow.
    Mlflow support Model Tracking, Experiment Tracking, and Model Registry.
    It can be hosted on-premises or in all the major cloud provider or with databricks also.

    For changing default logging Behaviour you can change mlflow environment variables:
        https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html

    For more information, please refer to:
    https://mlflow.org/docs/latest/introduction/index.html
    """
    def __init__(self):
        if not self.is_required_library_available():
            raise RuntimeError("MLflow Logging requires mlflow and python-dotenv to be installed. Run `pip install mlflow python-dotenv`." )

        import mlflow
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv())
        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH
        self._initialized = False
        self._auto_end_run = False
        self.best_ckpt_upload_pending = False
        self._ml_flow = mlflow

    def is_required_library_available(self):
        "check if required libraries are available."
        dotenv_availaible = importlib.util.find_spec("dotenv") is not None
        mlflow_available = importlib.util.find_spec("mlflow") is not None
        return dotenv_availaible and mlflow_available

    def flatten_dict(self, d: MutableMapping, parent_key: str = "", delimiter: str = "."):
        """Flatten a nested dict into a single level dict."""

        def _flatten_dict(d, parent_key="", delimiter="."):
            for k, v in d.items():
                key = str(parent_key) + delimiter + str(k) if parent_key else k
                if v and isinstance(v, MutableMapping):
                    yield from self.flatten_dict(v, key, delimiter=delimiter).items()
                else:
                    yield key, v

        return dict(_flatten_dict(d, parent_key, delimiter))

    def setup(self, trainer):
        """
        Setup the optional MLflow integration.
        - rank (int):
            0 only for main thread so publish result only from that

        Environment:
        - **YOLOX_MLFLOW_LOG_MODEL_ARTIFACTS** (`str`, *optional*, defaults to `False`):
            Whether to use MLflow `.log_artifact()` facility to log artifacts. This only makes sense if logging to a
            remote server, e.g. s3 or GCS. If set to `True` or *1*, will copy each checkpoints on each save in
            [`TrainingArguments`]'s `output_dir` to the local or remote artifact storage. Using it without a remote
            storage will just copy the files to your artifact location.
        - **YOLOX_MLFLOW_LOG_MODEL_PER_n_EPOCHS** (`int`, *optional*, defaults to 30):
            If ``YOLOX_MLFLOW_LOG_MODEL_ARTIFACTS`` is enabled then Log model checkpoints after every n epochs. Default is 30.
            ``best_ckpt.pth`` will be updated after `n` epochs if it has been updated during last `n`  epochs.
        - **YOLOX_MLFLOW_LOG_Nth_EPOCH_MODELS** (`str`, *optional*, defaults to `False`):
            Whether to log the ``epoch_n_ckpt.pth`` models along with best_ckpt.pth model after every `n` epochs. If set to `True` or *1*,
            will log ``epoch_n_ckpt.pth`` along with ``best_ckpt.pth`` and as mlflow artifacts in different folders.
        - **YOLOX_MLFLOW_RUN_NAME** (`str`, *optional*, defaults to random name):
            Name of new run. Used only when ``run_id`` is unspecified. If a new run is
            created and ``run_name`` is not specified, a random name will be generated for the run.
        - **YOLOX_MLFLOW_FLATTEN_PARAMS** (`str`, *optional*, defaults to `False`):
            Whether to flatten the parameters dictionary before logging.
        - **MLFLOW_TRACKING_URI** (`str`, *optional*):
            Whether to store runs at a specific path or remote server. Unset by default, which skips setting the
            tracking URI entirely.
        - **MLFLOW_EXPERIMENT_NAME** (`str`, *optional*, defaults to `None`):
            Whether to use an MLflow experiment_name under which to launch the run. Default to `None` which will point
            to the `Default` experiment in MLflow. Otherwise, it is a case-sensitive name of the experiment to be
            activated. If an experiment with this name does not exist, a new experiment with this name is created.
        - **MLFLOW_TAGS** (`str`, *optional*):
            A string dump of a dictionary of key/value pair to be added to the MLflow run as tags. Example:
            `os.environ['MLFLOW_TAGS']='{"release.candidate": "RC1", "release.version": "2.2.0"}'`.
        - **MLFLOW_NESTED_RUN** (`str`, *optional*):
            Whether to use MLflow nested runs. If set to `True` or *1*, will create a nested run inside the current
            run.
        - **MLFLOW_RUN_ID** (`str`, *optional*):
            Allow to reattach to an existing run which can be useful when resuming training from a checkpoint. When
            `MLFLOW_RUN_ID` environment variable is set, `start_run` attempts to resume a run with the specified run ID
            and other parameters are ignored.
        - Other MLflow environment variables: For changing default logging Behaviour refer mlflow environment variables:
        https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html
        - Setup ``Databricks`` integration with MLflow: Provide these two environment variables:
            DATABRICKS_HOST="https://adb-4273978218682429.9.azuredatabricks.net"
            DATABRICKS_TOKEN="dapixxxxxxxxxxxxx"
        """
        args = trainer.args
        self._tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
        self._experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", None)

        self._mlflow_log_artifacts = os.getenv("YOLOX_MLFLOW_LOG_MODEL_ARTIFACTS", "False").upper() in ENV_VARS_TRUE_VALUES
        self._mlflow_log_model_per_n_epochs = int(os.getenv("YOLOX_MLFLOW_LOG_MODEL_PER_n_EPOCHS", 30))
        self._mlflow_log_nth_epoch_models = os.getenv("YOLOX_MLFLOW_LOG_Nth_EPOCH_MODELS", "False").upper() in ENV_VARS_TRUE_VALUES
        self.run_name = os.getenv("YOLOX_MLFLOW_RUN_NAME", None)
        self.run_name = None if len(self.run_name.strip())==0 else self.run_name
        self._flatten_params = os.getenv("YOLOX_MLFLOW_FLATTEN_PARAMS", "FALSE").upper() in ENV_VARS_TRUE_VALUES

        self._nested_run = os.getenv("MLFLOW_NESTED_RUN", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._run_id = os.getenv("MLFLOW_RUN_ID", None)


        # "synchronous" flag is only available with mlflow version >= 2.8.0
        # https://github.com/mlflow/mlflow/pull/9705
        # https://github.com/mlflow/mlflow/releases/tag/v2.8.0
        self._async_log = packaging.version.parse(self._ml_flow.__version__) >= packaging.version.parse("2.8.0")

        logger.debug(
            f"MLflow experiment_name={self._experiment_name}, run_name={self.run_name}, nested={self._nested_run},"
            f" tags={self._nested_run}, tracking_uri={self._tracking_uri}"
        )
        rank = trainer.rank
        if rank==0:
            if not self._ml_flow.is_tracking_uri_set():
                if self._tracking_uri:
                    self._ml_flow.set_tracking_uri(self._tracking_uri)
                    logger.debug(f"MLflow tracking URI is set to {self._tracking_uri}")
                else:
                    logger.debug(
                        "Environment variable `MLFLOW_TRACKING_URI` is not provided and therefore will not be"
                        " explicitly set."
                    )
            else:
                logger.debug(f"MLflow tracking URI is set to {self._ml_flow.get_tracking_uri()}")

            if self._ml_flow.active_run() is None or self._nested_run or self._run_id:
                if self._experiment_name:
                    # Use of set_experiment() ensure that Experiment is created if not exists
                    self._ml_flow.set_experiment(self._experiment_name)
                self._ml_flow.start_run(run_name=self.run_name, nested=self._nested_run)
                logger.debug(f"MLflow run started with run_id={self._ml_flow.active_run().info.run_id}")
                self._auto_end_run = True
                self._initialized = True
            # filters these params from args
            keys = ['experiment_name', 'batch_size', 'exp_file', 'resume', 'ckpt', 'start_epoch', 'num_machines',  'fp16',  'logger']
            combined_dict = {k: v for k, v in vars(args).items() if k in keys}
            if trainer is not None:
                exp_dict = self.convert_exp_todict(trainer.exp)
                combined_dict = {**exp_dict, **combined_dict}
            self.log_params_mlflow(rank, combined_dict)
            mlflow_tags = os.getenv("MLFLOW_TAGS", None)
            if mlflow_tags:
                mlflow_tags = json.loads(mlflow_tags)
                self._ml_flow.set_tags(mlflow_tags)

    def log_params_mlflow(self, rank, params_dict, **kwargs):
        """
        Log hyperparameters to MLflow. MLflow's log_param() only accepts values no longer than 250 characters.
        no overwriting of existing parameters is allowed by default from mlflow.

        Input:
            rank: int 0 if it is main thread
            params_dict: dict of hyperparameters

        """

        if rank ==0:
            params_dict = self.flatten_dict(params_dict) if self._flatten_params else params_dict
            # remove params that are too long for MLflow
            for name, value in list(params_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f'Trainer is attempting to log a value of "{value}" for key "{name}" as a parameter. MLflow\'s'
                        " log_param() only accepts values no longer than 250 characters so we dropped this attribute."
                        " You can use `MLFLOW_FLATTEN_PARAMS` environment variable to flatten the parameters and"
                        " avoid this message."
                    )
                    del params_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(params_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                if self._async_log:
                    self._ml_flow.log_params(
                        dict(combined_dict_items[i: i + self._MAX_PARAMS_TAGS_PER_BATCH]), synchronous=False
                    )
                else:
                    self._ml_flow.log_params(dict(combined_dict_items[i: i + self._MAX_PARAMS_TAGS_PER_BATCH]))

    def convert_exp_todict(self, exp):
        """
        Convert the experiment object to dictionary for required parameter only

        Input:
            exp: Experiment object
        Output:
            exp_dict: dict of experiment parameters

        """
        filter_keys = ['max_epoch',  'num_classes', 'input_size', 'output_dir', 'data_dir', 'train_ann', 'val_ann', 'test_ann', 'test_conf', 'nmsthre']
        exp_dict = {k: v for k, v in exp.__dict__.items() if not k.startswith("__") and k in filter_keys}
        return exp_dict

    def on_log(self, trainer, logs,  **kwargs):
        """
        Log metrics to MLflow.

        Input:
            trainer: Trainer class object
            logs: dict of metrics

        """
        rank = trainer.rank
        step = trainer.progress_in_iter
        if not self._initialized:
            self.setup(trainer)
        if rank==0: # master thread only
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    metrics[k] = v.item()
                else:
                    logger.warning(
                        f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                        "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                    )

            if self._async_log:
                self._ml_flow.log_metrics(metrics=metrics, step=step, synchronous=False)
            else:
                self._ml_flow.log_metrics(metrics=metrics, step=step)

    def on_train_end(self, trainer, **kwargs):
        """
        mlflow logging action to take when training ends:

        1. log the training log file
        2. publish the latest best model to model_registry if it is allowed in config file
        3. close the mlfow run


        """
        if self._initialized and trainer.rank==0:
            self.save_log_file(trainer)
            if self.best_ckpt_upload_pending:
                model_file_name = "best_ckpt"
                mlflow_out_dir = f"{trainer.args.experiment_name}/{model_file_name}"
                artifact_path = os.path.join(trainer.file_name, f"{model_file_name}.pth")
                self.mlflow_save_pyfunc_model(trainer, artifact_path,
                                              mlflow_out_dir)  # regster model only after training ends
            if self._auto_end_run and self._ml_flow.active_run():
                self._ml_flow.end_run()

    def save_log_file(self, trainer):
        """
        Save the training log file to mlflow artifact path

        """
        log_file_path = os.path.join(trainer.file_name, "train_log.txt")
        mlflow_out_dir = f"{trainer.args.experiment_name}"
        logger.info(f"Logging logfile: {log_file_path}  in mlflow artifact path: {mlflow_out_dir}.")
        self._ml_flow.log_artifact(log_file_path, mlflow_out_dir)

    def save_checkpoints(self, trainer, update_best_ckpt):
        """
        Save the model checkpoints to mlflow artifact path
        Input:
            trainer: Trainer class object
            update_best_ckpt: bool to show if best_ckpt was updated
        """
        if trainer.rank == 0 and self._mlflow_log_artifacts :
            if update_best_ckpt:  # keep this in memory to upload model only at every upload frequency
                self.best_ckpt_upload_pending = True
            if ((trainer.epoch+1) % self._mlflow_log_model_per_n_epochs) == 0:
                self.save_log_file(trainer)
                if self.best_ckpt_upload_pending:
                    model_file_name = "best_ckpt"
                    mlflow_out_dir = f"{trainer.args.experiment_name}/{model_file_name}"
                    artifact_path = os.path.join(trainer.file_name, f"{model_file_name}.pth")
                    self.mlflow_save_pyfunc_model(trainer, artifact_path, mlflow_out_dir)
                    self.best_ckpt_upload_pending = False
                if self._mlflow_log_nth_epoch_models and trainer.exp.save_history_ckpt:
                    model_file_name = f"epoch_{trainer.epoch + 1}_ckpt"
                    mlflow_out_dir = f"{trainer.args.experiment_name}/{model_file_name}"
                    artifact_path = os.path.join(trainer.file_name, f"{model_file_name}.pth")
                    self.mlflow_save_pyfunc_model(trainer, artifact_path, mlflow_out_dir)

    def mlflow_save_pyfunc_model(self, trainer, artifact_path, mlflow_out_dir, **kwargs):
        """
        This will send the given model to mlflow server if HF_MLFLOW_LOG_ARTIFACTS is true
            - optionally publish to model registry if allowed in config file

        """
        if self._initialized and trainer.rank==0 and self._mlflow_log_artifacts:
            metadata = {
                "epoch": trainer.epoch + 1,
                "input_size": trainer.input_size,
                'start_ckpt': trainer.args.ckpt,
                'exp_file': trainer.args.exp_file,
                "best_ap": float(trainer.best_ap)
            }
            logger.info(f"Logging checkpoint {artifact_path} artifacts in mlflow artifact path: {mlflow_out_dir}. This may take time.")
            if os.path.exists(artifact_path):
                self._ml_flow.pyfunc.log_model(
                    mlflow_out_dir,
                    artifacts={"model_path": artifact_path},
                    python_model=self._ml_flow.pyfunc.PythonModel(),
                    metadata=metadata
                )

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if (
            self._auto_end_run
            and callable(getattr(self._ml_flow, "active_run", None))
            and self._ml_flow.active_run() is not None
        ):
            self._ml_flow.end_run()
