import argparse
from wandb_utils import WandbLogger


def create_dataset_artifact(opt): -> None:
    """Initializes the WandbLogger and creates a dataset artifact.

    Args:
        opt ([type]): Argparse options.
    """
    logger = WandbLogger(opt, None)  # TODO: return value unused
    logger.create_dataset_artifact(opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=str,
        default="datasets/COCO/train2017",
        help="training path",
    )
    parser.add_argument(
        "--val-path", type=str, default="datasets/COCO/val2017", help="validation path"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="train as single-class dataset"
    )
    parser.add_argument(
        "--project", type=str, default="YOLOX", help="name of W&B Project"
    )
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument(
        "--name", type=str, default="COCO_data_log", help="name of W&B run"
    )
    parser.add_argument(
        "--job", type=str, default="dataset", help="type of job"
    )

    opt = parser.parse_args()
    opt.resume = False  # Explicitly disallow resume check for dataset upload job

    create_dataset_artifact(opt)
