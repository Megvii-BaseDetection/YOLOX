#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Usage example:
    import torch
    model = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_s")
    model = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_custom",
                           exp_path="exp.py", ckpt_path="ckpt.pth")
"""
from yolox.utils.hub import ensure_installed

HUB_REQUIREMENTS = ["loguru", "torchvision", "thop", "opencv-python-headless==4.5.2.52", "tabulate"]
ensure_installed(HUB_REQUIREMENTS)

from yolox.models import (  # isort:skip  # noqa: F401, E402
    yolox_tiny,
    yolox_nano,
    yolox_s,
    yolox_m,
    yolox_l,
    yolox_x,
    yolov3,
    yolox_custom
)
