from typing import Iterable, TypedDict, Union

import numpy as np
import torch
from PIL.Image import Image

from yolox import data, utils
from yolox.exp import Exp, get_exp


class YoloxProcessor:
    exp: Exp
    threshold: float

    def __init__(
        self,
        model_name_or_exp: Union[str, Exp],
        threshold: float = 0.5,
    ):
        if not isinstance(model_name_or_exp, (str, Exp)):
            raise ValueError("model_name_or_exp must be a string or Exp")

        if isinstance(model_name_or_exp, str):
            self.exp = get_exp(exp_name=model_name_or_exp)
        else:
            self.exp = model_name_or_exp
        self.threshold = threshold

    def __call__(self, inputs: Iterable[Image]) -> torch.Tensor:
        return self.__images_to_tensor(inputs)

    def __images_to_tensor(self, images: Iterable[Image]) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        _val_transform = data.ValTransform(legacy=False)
        for image in images:
            # image = normalize_image_mode(image)
            image_transform, _ = _val_transform(np.array(image), None, self.exp.test_size)
            tensors.append(torch.from_numpy(image_transform))
        return torch.stack(tensors)

    def postprocess(self, images: Iterable[Image], tensor: torch.Tensor) -> list['Detections']:
        outputs: list[torch.Tensor] = utils.postprocess(tensor, self.exp.num_classes, self.exp.nmsthre, class_agnostic=False)
        results: list[Detections] = []
        for i, image in enumerate(images):
            ratio = min(self.exp.test_size[0] / image.height, self.exp.test_size[1] / image.width)
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
