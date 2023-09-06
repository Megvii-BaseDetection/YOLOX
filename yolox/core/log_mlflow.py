"""Mlflow logging module."""
import mlflow
from datetime import datetime


def logger_init(exp_name):
    """Initialize mlflow logger."""
    MLFLOW_SERVER = "http://192.168.189.67:8888"
    mlflow_exp_name = exp_name + " " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mlflow.set_tracking_uri(MLFLOW_SERVER)
    mlflow.set_experiment(mlflow_exp_name)
    mlflow.start_run()


def log_params_and_model(exp, args, optimizer, model):
    """Log params and model to mlflow."""
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
    mlflow.pytorch.log_model(model, "model")


def log_metrics(epoch_meter, epoch):
    """Log metrics to mlflow."""
    epoch_loss_meter = epoch_meter.get_filtered_meter("loss")
    for k, v in epoch_loss_meter.items():
        mlflow.log_metric(k, v.latest, step=epoch + 1)
    mlflow.log_metric("lr", epoch_meter["lr"].latest, step=epoch + 1)


def log_valid_metrics(m_ap, per_class_AP, per_class_AR, epoch):
    """Log mAP, AP per class and AR per class to mlflow."""
    for class_name, ap in per_class_AP.items():
        mlflow.log_metric(f"{class_name}_AP", ap, step=epoch + 1)
    for class_name, ar in per_class_AR.items():
        mlflow.log_metric(f"{class_name}_AR", ar, step=epoch + 1)
    mlflow.log_metric("mAP", m_ap * 100, step=epoch + 1)


def log_best_map_end_run(best_m_ap):
    """Log best ap and end run"""
    mlflow.log_param("best_mAP", best_m_ap * 100)
    mlflow.end_run()
