"""CPU/CUDA selection shared by the training and evaluation entry points."""

from loguru import logger

import torch


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "gpu":
        name = "cuda"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; install the cuda extra and check the NVIDIA driver")
    return device


def configure_device(args):
    device = get_device(args.device)
    args.device = device.type
    if device.type == "cpu":
        if args.num_machines != 1 or args.devices not in (None, 0, 1):
            raise ValueError("CPU execution supports one process; use --devices 1")
        if args.fp16:
            logger.warning("FP16 requires CUDA; using FP32 on CPU")
            args.fp16 = False
        if getattr(args, "occupy", False):
            logger.warning("GPU memory preallocation is disabled on CPU")
            args.occupy = False
        return 1
    count = torch.cuda.device_count() if args.devices is None else args.devices
    if not 1 <= count <= torch.cuda.device_count():
        raise ValueError("--devices must be between 1 and the available CUDA device count")
    return count
