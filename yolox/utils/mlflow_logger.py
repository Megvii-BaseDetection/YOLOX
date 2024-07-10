#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.
# Please read docs/mlflow_integration.md for more details.
"""
Logging training runs with hyperparameter, datasets and trained models to MlFlow.
Mlflow support Model Tracking, Experiment Tracking, and Model Registry.
It can be hosted on-premises or in all the major cloud provider or with databricks also.
Please read docs/mlflow_integration.md for more details.

For changing default logging Behaviour you can change mlflow environment variables:
    https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html

For more information, please refer to:
https://mlflow.org/docs/latest/introduction/index.html
"""
import importlib.metadata
import importlib.util
import json
import os
from collections.abc import MutableMapping
import packaging.version
from loguru import logger

import torch

from yolox.utils import is_main_process


class MlflowLogger:
    """
    Main Mlflow logging class to log hyperparameters, metrics, and models to Mlflow.
    """
    def __init__(self):
        if not self.is_required_library_available():
            raise RuntimeError(
                "MLflow Logging requires mlflow and python-dotenv to be installed. "
                "Run `pip install mlflow python-dotenv`.")

        import mlflow
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv())
        self.ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH
        self._initialized = False
        self._auto_end_run = False
        self.best_ckpt_upload_pending = False
        self._tracking_uri = None
        self._experiment_name = None
        self._mlflow_log_artifacts = None
        self._mlflow_log_model_per_n_epochs = None
        self._mlflow_log_nth_epoch_models = None
        self.run_name = None
        self._flatten_params = None
        self._nested_run = None
        self._run_id = None
        self._async_log = None
        self._ml_flow = mlflow

    def is_required_library_available(self):
        """
        check if required libraries are available.

        Args: None

        Returns:
            bool: True if required libraries are available, False otherwise.
        """
        dotenv_availaible = importlib.util.find_spec("dotenv") is not None
        mlflow_available = importlib.util.find_spec("mlflow") is not None
        return dotenv_availaible and mlflow_available

    def flatten_dict(self, d: MutableMapping, parent_key: str = "", delimiter: str = "."):
        """
        Flatten a nested dict into a single level dict.

        Args:
            d(MutableMapping): nested dictionary
            parent_key(str): parent key
            delimiter(str): delimiter to use

        Returns:
            flattened_dict(dict): flattened dictionary

        """

        def _flatten_dict(d, parent_key="", delimiter="."):
            for k, v in d.items():
                key = str(parent_key) + delimiter + str(k) if parent_key else k
                if v and isinstance(v, MutableMapping):
                    yield from self.flatten_dict(v, key, delimiter=delimiter).items()
                else:
                    yield key, v

        return dict(_flatten_dict(d, parent_key, delimiter))

    def setup(self, args, exp):
        """
        Set up the optional MLflow integration.

        Args:
            args(dict): training args dictionary
            exp(dict): Experiment related hyperparameters

        Returns:
            None

        Environment:
        - **YOLOX_MLFLOW_LOG_MODEL_ARTIFACTS** (`str`, *optional*, defaults to `False`):
            Whether to use MLflow `.log_artifact()` facility to log artifacts. This only makes
            sense if logging to a remote server, e.g. s3 or GCS. If set to `True` or *1*,
            will copy each check-points on each save in [`TrainingArguments`]'s `output_dir` to the
            local or remote artifact storage. Using it without a remote storage will just copy the
            files to your artifact location.
        - **YOLOX_MLFLOW_LOG_MODEL_PER_n_EPOCHS** (`int`, *optional*, defaults to 30):
            If ``YOLOX_MLFLOW_LOG_MODEL_ARTIFACTS`` is enabled then Log model checkpoints after
            every n epochs. Default is 30. ``best_ckpt.pth`` will be updated after `n` epochs if
            it has been updated during last `n`  epochs.
        - **YOLOX_MLFLOW_LOG_Nth_EPOCH_MODELS** (`str`, *optional*, defaults to `False`):
            Whether to log the ``epoch_n_ckpt.pth`` models along with best_ckpt.pth model after
             every `n` epoch as per YOLOX_MLFLOW_LOG_MODEL_PER_n_EPOCHS.
             If set to `True` or *1*, will log ``epoch_n_ckpt.pth`` along with
             ``best_ckpt.pth`` and as mlflow artifacts in different folders.
        - **YOLOX_MLFLOW_RUN_NAME** (`str`, *optional*, defaults to random name):
            Name of new run. Used only when ``run_id`` is unspecified. If a new run is
            created and ``run_name`` is not specified, a random name will be generated for the run.
        - **YOLOX_MLFLOW_FLATTEN_PARAMS** (`str`, *optional*, defaults to `False`):
            Whether to flatten the parameters dictionary before logging.
        - **MLFLOW_TRACKING_URI** (`str`, *optional*):
            Whether to store runs at a specific path or remote server. Unset by default, which
            skips setting the tracking URI entirely.
        - **MLFLOW_EXPERIMENT_NAME** (`str`, *optional*, defaults to `None`):
            Whether to use an MLflow experiment_name under which to launch the run. Default to
            `None` which will point to the `Default` experiment in MLflow. Otherwise, it is a
            case-sensitive name of the experiment to be activated. If an experiment with this
            name does not exist, a new experiment with this name is created.
        - **MLFLOW_TAGS** (`str`, *optional*):
            A string dump of a dictionary of key/value pair to be added to the MLflow run as tags.
             Example: `os.environ['MLFLOW_TAGS']=
             '{"release.candidate": "RC1", "release.version": "2.2.0"}'`.
        - **MLFLOW_NESTED_RUN** (`str`, *optional*):
            Whether to use MLflow nested runs. If set to `True` or *1*, will create a nested run
            inside the current run.
        - **MLFLOW_RUN_ID** (`str`, *optional*):
            Allow to reattach to an existing run which can be useful when resuming training from a
             checkpoint. When `MLFLOW_RUN_ID` environment variable is set, `start_run` attempts
             to resume a run with the specified run ID and other parameters are ignored.
        - Other MLflow environment variables: For changing default logging Behaviour refer mlflow
            environment variables:
        https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html
        - Setup ``Databricks`` integration with MLflow: Provide these two environment variables:
            DATABRICKS_HOST="https://adb-4273978218682429.9.azuredatabricks.net"
            DATABRICKS_TOKEN="dapixxxxxxxxxxxxx"
        """
        self._tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
        self._experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", None)
        self._mlflow_log_artifacts = os.getenv("YOLOX_MLFLOW_LOG_MODEL_ARTIFACTS",
                                               "False").upper() in self.ENV_VARS_TRUE_VALUES
        self._mlflow_log_model_per_n_epochs = int(os.getenv(
            "YOLOX_MLFLOW_LOG_MODEL_PER_n_EPOCHS", 30))

        self._mlflow_log_nth_epoch_models = os.getenv("YOLOX_MLFLOW_LOG_Nth_EPOCH_MODELS",
                                                      "False").upper() in self.ENV_VARS_TRUE_VALUES
        self.run_name = os.getenv("YOLOX_MLFLOW_RUN_NAME", None)
        self.run_name = None if len(self.run_name.strip()) == 0 else self.run_name
        self._flatten_params = os.getenv("YOLOX_MLFLOW_FLATTEN_PARAMS",
                                         "FALSE").upper() in self.ENV_VARS_TRUE_VALUES
        self._nested_run = os.getenv("MLFLOW_NESTED_RUN",
                                     "FALSE").upper() in self.ENV_VARS_TRUE_VALUES
        self._run_id = os.getenv("MLFLOW_RUN_ID", None)

        # "synchronous" flag is only available with mlflow version >= 2.8.0
        # https://github.com/mlflow/mlflow/pull/9705
        # https://github.com/mlflow/mlflow/releases/tag/v2.8.0
        self._async_log = packaging.version.parse(
            self._ml_flow.__version__) >= packaging.version.parse("2.8.0")

        logger.debug(
            f"MLflow experiment_name={self._experiment_name}, run_name={self.run_name}, "
            f"nested={self._nested_run}, tags={self._nested_run}, tracking_uri={self._tracking_uri}"
        )
        if is_main_process():
            if not self._ml_flow.is_tracking_uri_set():
                if self._tracking_uri:
                    self._ml_flow.set_tracking_uri(self._tracking_uri)
                    logger.debug(f"MLflow tracking URI is set to {self._tracking_uri}")
                else:
                    logger.debug(
                        "Environment variable `MLFLOW_TRACKING_URI` is not provided and therefore"
                        " will not be explicitly set."
                    )
            else:
                logger.debug(f"MLflow tracking URI is set to {self._ml_flow.get_tracking_uri()}")

            if self._ml_flow.active_run() is None or self._nested_run or self._run_id:
                if self._experiment_name:
                    # Use of set_experiment() ensure that Experiment is created if not exists
                    self._ml_flow.set_experiment(self._experiment_name)
                self._ml_flow.start_run(run_name=self.run_name, nested=self._nested_run)
                logger.debug(
                    f"MLflow run started with run_id={self._ml_flow.active_run().info.run_id}")
                self._auto_end_run = True
                self._initialized = True
            # filters these params from args
            keys = ['experiment_name', 'batch_size', 'exp_file', 'resume', 'ckpt', 'start_epoch',
                    'num_machines', 'fp16', 'logger']
            combined_dict = {k: v for k, v in vars(args).items() if k in keys}
            if exp is not None:
                exp_dict = self.convert_exp_todict(exp)
                combined_dict = {**exp_dict, **combined_dict}
            self.log_params_mlflow(combined_dict)
            mlflow_tags = os.getenv("MLFLOW_TAGS", None)
            if mlflow_tags:
                mlflow_tags = json.loads(mlflow_tags)
                self._ml_flow.set_tags(mlflow_tags)

    def log_params_mlflow(self, params_dict):
        """
        Log hyperparameters to MLflow.
        MLflow's log_param() only accepts values no longer than 250 characters.
        No overwriting of existing parameters is allowed by default from mlflow.

        Args:
            params_dict(dict): dict of hyperparameters

        Returns:
            None
        """
        if is_main_process():
            params_dict = self.flatten_dict(params_dict) if self._flatten_params else params_dict
            # remove params that are too long for MLflow
            for name, value in list(params_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f'Trainer is attempting to log a value of "{value}" for key "{name}" as a '
                        f'parameter. MLflow\'s log_param() only accepts values no longer than 250 '
                        f'characters so we dropped this attribute. You can use '
                        f'`MLFLOW_FLATTEN_PARAMS` environment variable to flatten the parameters '
                        f'and avoid this message.'
                    )
                    del params_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(params_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                if self._async_log:
                    self._ml_flow.log_params(
                        dict(combined_dict_items[i: i + self._MAX_PARAMS_TAGS_PER_BATCH]),
                        synchronous=False
                    )
                else:
                    self._ml_flow.log_params(
                        dict(combined_dict_items[i: i + self._MAX_PARAMS_TAGS_PER_BATCH])
                    )

    def convert_exp_todict(self, exp):
        """
        Convert the experiment object to dictionary for required parameter only

        Args:
            exp(dict): Experiment object

        Returns:
            exp_dict(dict): dict of experiment parameters

        """
        filter_keys = ['max_epoch', 'num_classes', 'input_size', 'output_dir',
                       'data_dir', 'train_ann', 'val_ann', 'test_ann',
                       'test_conf', 'nmsthre']
        exp_dict = {k: v for k, v in exp.__dict__.items()
                    if not k.startswith("__") and k in filter_keys}
        return exp_dict

    def on_log(self, args, exp, step, logs):
        """
        Log metrics to MLflow.

        Args:
            args(dict): training args dictionary
            exp(dict): Experiment related hyperparameters
            step(int): current training step
            logs(dict): dictionary of logs to be logged

        Returns:
            None
        """
        # step = trainer.progress_in_iter
        if not self._initialized:
            self.setup(args, exp)
        if is_main_process():  # master thread only
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    metrics[k] = v.item()
                else:
                    logger.warning(
                        f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key '
                        f'"{k}" as a metric. MLflow log_metric() only accepts float and int types '
                        f'so we dropped this attribute.'
                    )

            if self._async_log:
                self._ml_flow.log_metrics(metrics=metrics, step=step, synchronous=False)
            else:
                self._ml_flow.log_metrics(metrics=metrics, step=step)

    def on_train_end(self, args, file_name, metadata):
        """
        Mlflow logging action to take when training ends:
            1. log the training log file
            2. publish the latest best model to model_registry if it is allowed in config file
            3. close the mlfow run

        Args:
            args(dict): training args dictionary
            file_name(str): output directory
            metadata(dict): model related metadata

        Returns:
            None
        """
        if is_main_process() and self._initialized:
            self.save_log_file(args, file_name)
            if self.best_ckpt_upload_pending:
                model_file_name = "best_ckpt"
                mlflow_out_dir = f"{args.experiment_name}/{model_file_name}"
                artifact_path = os.path.join(file_name, f"{model_file_name}.pth")
                self.mlflow_save_pyfunc_model(metadata, artifact_path, mlflow_out_dir)
            if self._auto_end_run and self._ml_flow.active_run():
                self._ml_flow.end_run()

    def save_log_file(self, args, file_name):
        """
        Save the training log file to mlflow artifact path
        Args:
            args(dict): training args dictionary
            file_name(str): output directory

        Returns:
            None
        """
        log_file_path = os.path.join(file_name, "train_log.txt")
        mlflow_out_dir = f"{args.experiment_name}"
        logger.info(f"Logging logfile: {log_file_path} in mlflow artifact path: {mlflow_out_dir}.")
        self._ml_flow.log_artifact(log_file_path, mlflow_out_dir)

    def save_checkpoints(self, args, exp, file_name, epoch, metadata, update_best_ckpt):
        """
        Save the model checkpoints to mlflow artifact path
        if save_history_ckpt is enabled then

        Args:
            args(dict): training args dictionary
            exp(dict): Experiment related hyperparameters
            file_name(str): output directory
            epoch(int): current epoch
            metadata(dict): model related metadata
            update_best_ckpt(bool): bool to show if best_ckpt was updated

        Returns:
            None
        """
        if is_main_process() and self._mlflow_log_artifacts:
            if update_best_ckpt:
                self.best_ckpt_upload_pending = True
            if ((epoch + 1) % self._mlflow_log_model_per_n_epochs) == 0:
                self.save_log_file(args, file_name)
                if self.best_ckpt_upload_pending:
                    model_file_name = "best_ckpt"
                    mlflow_out_dir = f"{args.experiment_name}/{model_file_name}"
                    artifact_path = os.path.join(file_name, f"{model_file_name}.pth")
                    self.mlflow_save_pyfunc_model(metadata, artifact_path, mlflow_out_dir)
                    self.best_ckpt_upload_pending = False
                if self._mlflow_log_nth_epoch_models and exp.save_history_ckpt:
                    model_file_name = f"epoch_{epoch + 1}_ckpt"
                    mlflow_out_dir = f"{args.experiment_name}/hist_epochs/{model_file_name}"
                    artifact_path = os.path.join(file_name, f"{model_file_name}.pth")
                    self.mlflow_save_pyfunc_model(metadata, artifact_path, mlflow_out_dir)

    def mlflow_save_pyfunc_model(self, metadata, artifact_path, mlflow_out_dir):
        """
        This will send the given model to mlflow server if HF_MLFLOW_LOG_ARTIFACTS is true
            - optionally publish to model registry if allowed in config file

        Args:
            metadata(dict): model related metadata
            artifact_path(str): model checkpoint path
            mlflow_out_dir(str): mlflow artifact path

        Returns:
            None
        """
        if is_main_process() and self._initialized and self._mlflow_log_artifacts:
            logger.info(
                f"Logging checkpoint {artifact_path} artifacts in mlflow artifact path: "
                f"{mlflow_out_dir}. This may take time.")
            if os.path.exists(artifact_path):
                self._ml_flow.pyfunc.log_model(
                    mlflow_out_dir,
                    artifacts={"model_path": artifact_path},
                    python_model=self._ml_flow.pyfunc.PythonModel(),
                    metadata=metadata
                )

    def __del__(self):
        """
        if the previous run is not terminated correctly, the fluent API will
        not let you start a new run before the previous one is killed

        Args: None
        Return: None
        """
        if (
                self._auto_end_run
                and callable(getattr(self._ml_flow, "active_run", None))
                and self._ml_flow.active_run() is not None
        ):
            self._ml_flow.end_run()
