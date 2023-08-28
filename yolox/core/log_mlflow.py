"""Mlflow logging module."""
import mlflow
from datetime import datetime


def mlflow_logger_init(exp_name):
    MLFLOW_SERVER = "http://192.168.189.67:8888"
    mlflow_exp_name = exp_name + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mlflow.set_tracking_uri(MLFLOW_SERVER)
    mlflow.set_experiment(mlflow_exp_name)
    mlflow.start_run()


def mlflow_log_params(exp, args, optimizer, data_type):
    mlflow.log_param("epochs", exp.max_epoch)
    mlflow.log_param("num_classes", exp.num_classes)
    mlflow.log_param("depth", exp.depth)
    mlflow.log_param("width", exp.width)
    mlflow.log_param("input_size", exp.input_size)
    mlflow.log_param("optimizer", optimizer.__class__.__name__)
    mlflow.log_param("data_type", "float16" if args.fp16 else "float32")
    mlflow.log_param("data_dir", exp.data_dir)
    mlflow.log_param("min_lr_ratio", exp.min_lr_ratio)
    
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("basic_lr_per_img", exp.basic_lr_per_img)



def mlflow_log_end_run():
    mlflow.end_run()
