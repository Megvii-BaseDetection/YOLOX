# Copyright (c) Megvii Inc. All rights reserved.

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Union

from PIL import Image
import torch
import torch.nn as nn

from yolox.config import YoloxConfig
from yolox.models.processor import Detections, YoloxProcessor

from .yolo_head import YoloxHead
from .yolo_pafpn import YoloPafpn

HOME = Path(os.environ.get('YOLOX_HOME', str(Path.home() / '.cache' / 'yolox')))

class Yolox:
    module: YoloxModule
    processor: YoloxProcessor

    def __init__(self, module: YoloxModule, processor: YoloxProcessor):
        self.module = module
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        config: Optional[YoloxConfig] = None,
        device: str = 'cpu',
    ) -> Yolox:
        module = YoloxModule.from_pretrained(pretrained_model_name_or_path, config, device)
        processor = YoloxProcessor(config or pretrained_model_name_or_path)
        return cls(module, processor)

    def __call__(self, inputs: Iterable[Union[Image.Image, str, os.PathLike]], threshold: float = 0.5) -> list[Detections]:
        if isinstance(inputs, torch.Tensor):
            # For backward compatibility (deprecated call pattern; use YoloxModule instead)
            return self.module(inputs)

        images: list[Image.Image] = [
            image if isinstance(image, Image.Image) else Image.open(image)
            for image in inputs
        ]
        tensor = self.processor(images)
        output = self.module(tensor)
        return self.processor.postprocess(images, output, threshold=threshold)


class YoloxModule(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone: Optional[YoloPafpn] = None, head: Optional[YoloxHead] = None):
        super().__init__()
        if backbone is None:
            backbone = YoloPafpn()
        if head is None:
            head = YoloxHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        config: Optional[YoloxConfig] = None,
        device: str = 'cpu',
    ) -> YoloxModule:
        path = str(pretrained_model_name_or_path)
        if os.path.isfile(path):
            if config is None:
                raise ValueError('config must be provided when loading model from a file')
        else:
            config = YoloxConfig.get_named_config(path)
            if config is None:
                raise ValueError(f'Unknown model: {pretrained_model_name_or_path}')
            path = cls.__cached_pretrained_weights(path)
        model = config.get_model().to(device)
        model.eval()
        model.head.training = False
        model.training = False
        weights = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(weights['model'])
        return model

    @classmethod
    def __cached_pretrained_weights(cls, model_id: str) -> str:
        weights_dir = HOME / 'weights'
        weights_dir.mkdir(exist_ok=True, parents=True)
        weights_file = weights_dir / f'{model_id}.pth'
        if not weights_file.exists():
            weights_url = f'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{model_id}.pth'
            urllib.request.urlretrieve(weights_url, f'{weights_file}.tmp')
            os.rename(f'{weights_file}.tmp', weights_file)
        return str(weights_file)
