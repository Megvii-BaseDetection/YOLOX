"""Mlflow logging module."""
import mlflow
from datetime import datetime


def mlflow_logger_init(exp_name):
    MLFLOW_SERVER = "http://192.168.189.67:8888"
    mlflow_exp_name = exp_name + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mlflow.set_tracking_uri(MLFLOW_SERVER)
    mlflow.set_experiment(mlflow_exp_name)
    mlflow.start_run()


def mlflow_log_params(exp, args):
    mlflow.log_param("Epochs", exp.max_epoch)
    mlflow.log_param("Num_classes", exp.num_classes)
    mlflow.log_param("Depth", exp.depth)
    mlflow.log_param("Width", exp.width)
    mlflow.log_param("Batch size", args.batch_size)
    mlflow.log_param("basic_lr_per_img", exp.basic_lr_per_img)


def mlflow_log_end_run():
    mlflow.end_run()
