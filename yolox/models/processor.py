from __future__ import annotations

from typing import Iterable, TypedDict, Union

import numpy as np
import torch
from PIL.Image import Image

from yolox import data, utils
from yolox.config import YoloxConfig


class YoloxProcessor:
    config: YoloxConfig

    def __init__(
        self,
        model_name_or_config: Union[str, YoloxConfig],
    ):
        if isinstance(model_name_or_config, str):
            self.config = YoloxConfig.get_named_config(model_name_or_config)
        elif isinstance(model_name_or_config, YoloxConfig):
            self.config = model_name_or_config
        else:
            raise ValueError("model_name_or_config must be a string or YoloxConfig")

    def __call__(self, inputs: Iterable[Image]) -> torch.Tensor:
        return self.__images_to_tensor(inputs)

    def __images_to_tensor(self, images: Iterable[Image]) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        _val_transform = data.ValTransform(legacy=False)
        for image in images:
            # image = normalize_image_mode(image)
            image_transform, _ = _val_transform(np.array(image), None, self.config.test_size)
            tensors.append(torch.from_numpy(image_transform))
        return torch.stack(tensors)

    def postprocess(self, images: Iterable[Image], tensor: torch.Tensor, threshold: float = 0.5) -> list[Detections]:
        outputs: list[torch.Tensor] = utils.postprocess(tensor, self.config.num_classes, threshold, self.config.nmsthre, class_agnostic=False)
        results: list[Detections] = []
        for i, image in enumerate(images):
            ratio = min(self.config.test_size[0] / image.height, self.config.test_size[1] / image.width)
            if outputs[i] is None:
                results.append(Detections(bboxes=[], scores=[], labels=[]))
            else:
                results.append(
                    Detections(
                        bboxes=[tuple((output[:4] / ratio).tolist()) for output in outputs[i]],
                        scores=[output[4].item() * output[5].item() for output in outputs[i]],
                        labels=[int(output[6]) for output in outputs[i]],
                    )
                )
        return results


class Detections(TypedDict):
    bboxes: list[tuple[float, float, float, float]]
    scores: list[float]
    labels: list[int]
