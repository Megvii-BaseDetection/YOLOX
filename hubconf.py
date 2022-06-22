#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Usage example:
    import torch
    model = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_s")
"""
from tools.hub import ensure_installed

HUB_REQUIREMENTS = ["loguru", "numpy", "torchvision", "thop", "onnx", "opencv_python", "tabulate"]
ensure_installed(HUB_REQUIREMENTS)

from yolox.models import (  # isort:skip  # noqa: F401, E402
    yolox_tiny,
    yolox_nano,
    yolox_s,
    yolox_m,
    yolox_l,
    yolox_x,
    yolov3,
)
