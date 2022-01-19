#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import megengine as mge
import megengine.module as M

from models.yolo_fpn import YOLOFPN
from models.yolo_head import YOLOXHead
from models.yolo_pafpn import YOLOPAFPN
from models.yolox import YOLOX


def build_yolox(name="yolox-s"):
    num_classes = 80

    # value meaning: depth, width
    param_dict = {
        "yolox-nano": (0.33, 0.25),
        "yolox-tiny": (0.33, 0.375),
        "yolox-s": (0.33, 0.50),
        "yolox-m": (0.67, 0.75),
        "yolox-l": (1.0, 1.0),
        "yolox-x": (1.33, 1.25),
    }
    if name == "yolov3":
        depth = 1.0
        width = 1.0
        backbone = YOLOFPN()
        head = YOLOXHead(num_classes, width, in_channels=[128, 256, 512], act="lrelu")
        model = YOLOX(backbone, head)
    else:
        assert name in param_dict
        kwargs = {}
        depth, width = param_dict[name]
        if name == "yolox-nano":
            kwargs["depthwise"] = True
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels, **kwargs)
        head = YOLOXHead(num_classes, width, in_channels=in_channels, **kwargs)
        model = YOLOX(backbone, head)

    for m in model.modules():
        if isinstance(m, M.BatchNorm2d):
            m.eps = 1e-3

    return model


def build_and_load(weight_file, name="yolox-s"):
    model = build_yolox(name)
    model_weights = mge.load(weight_file)
    model.load_state_dict(model_weights, strict=False)
    return model
