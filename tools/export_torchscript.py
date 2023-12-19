#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger
from yolox_deploy import DeployModule
import torch
from yolox.data.data_augment import ValTransform
import cv2
from yolox.exp import get_exp
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser("YOLOX torchscript deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.torchscript.pt", help="output name of models"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.head.decode_in_inference = args.decode_in_inference
    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    mod = torch.jit.trace(model, dummy_input)
    mod.save(args.output_name)
    logger.info("generated torchscript model named {}".format(args.output_name))


def load_image(file_path: str, raise_on_error: bool = True) -> np.ndarray:
    """Load image from the file and convert color space to RGB."""
    image = cv2.imread(file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_image_shape(image: np.ndarray) -> tuple[int, int]:
    """Return image shape (width, height) for given image."""
    height, width = image.shape[:2]
    return width, height

def letterbox_image(image: np.ndarray, new_size: tuple[int, int], boxes: np.ndarray | None = None) \
        -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Resize image with unchanged aspect ratio using padding. Padded areas are filled with gray (128, 128, 128) color.

    :param image: Opencv RGB image.
    :param new_size: Expected (width, height) size after resizing.
    :param boxes: Optional. If set, also boxes data is rescaled.
    :return: Resized image or tuple (resized image, resized boxes).
    """
    current_width, current_height = get_image_shape(image)
    final_width, final_height = new_size
    scale = min(final_width / current_width, final_height / current_height)
    scaled_width = int(current_width * scale)
    scaled_height = int(current_height * scale)
    image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
    if boxes is not None:
        if len(boxes) > 0:  # Check if array is not empty. This is different situation than boxes = None.
            boxes[:, :-1] = boxes[:, :-1] * scale
        return extend_image(image, new_size, boxes)
    return extend_image(image, new_size)

def extend_image(image: np.ndarray, size: tuple[int, int], boxes: np.ndarray | None = None, dx: int | None = None,
                 dy: int | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Extend image using padding to get expected size. Added areas are filled with gray (128, 128, 128) color.
    Assume that both current image dimensions are smaller or equal to the expected final size.

    :param image: Opencv RGB image.
    :param size: Expected (width, height) image size after padding.
    :param boxes: Optional. Bounding boxes coordinates. Shape (number of boxes, 4).
    :param dx: Optional. Padding on the left side. If not set, padding is centered.
    :param dy: Optional. Padding on the top side. If not set, padding is centered.
    :return: Final image or tuple (final image, final boxes).
    """
    current_width, current_height = get_image_shape(image)
    final_width, final_height = size
    assert final_width >= current_width
    assert final_height >= current_height
    dx = dx or (final_width - current_width) // 2
    dy = dy or (final_height - current_height) // 2
    new_image = np.ones((final_height, final_width, 3)) * 128
    new_image[dy: dy + current_height, dx: dx + current_width] = image
    if boxes is not None:
        if len(boxes) > 0:  # Check if array is not empty. This is different situation than boxes = None.
            boxes[:, 0:-1:2] += dx
            boxes[:, 1:-1:2] += dy
        return new_image, boxes
    return new_image


def get_tensor_image(image_path):
    test_size = (416, 416)
    img = cv2.imread(image_path)
    preproc = ValTransform(legacy=False)
    img, _ = preproc(img, None, test_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    return img.permute(0, 2, 3, 1)


@logger.catch
def args_main(exp_file, name, opts, decode_in_inference, batch_size, output_name, ckpt=None):
    exp = get_exp(exp_file, name)
    exp.merge(opts)

    model = exp.get_model()
    if ckpt is None:
        file_name = os.path.join(exp.output_dir, name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.head.decode_in_inference = decode_in_inference
    model_deploy = DeployModule(1, 0.3, 0.45, True, (416, 416))
    model_deploy.yolox = model
    model.eval()
    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(batch_size, 3, exp.test_size[0], exp.test_size[1], dtype=torch.float32)
    image_path = r'C:\Users\arusak\new zones 2\clean view\image.jpg'
    image = get_tensor_image(image_path)
    test_zone = torch.tensor([
        [
            446,
            149
        ],
        [
            501,
            160
        ],
        [
            501,
            265
        ],
        [
            482,
            444
        ],
        [
            429,
            431
        ],
        [
            399,
            544
        ],
        [
            453,
            556
        ],
        [
            439,
            604
        ],
        [
            381,
            594
        ],
        [
            377,
            678
        ],
        [
            310,
            667
        ],
        [
            298,
            570
        ],
        [
            298,
            438
        ],
        [
            205,
            418
        ],
        [
            213,
            247
        ],
        [
            362,
            246
        ],
        [
            364,
            157
        ]
    ], dtype=torch.float32)
    image = load_image(r'C:\Users\arusak\new zones 2\clean view\image.jpg')
    image = letterbox_image(image, (416, 416))
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image.tolist(), dtype=torch.float32)
    mod = torch.jit.trace(model_deploy, {"inputs": {"input": image, "zone": test_zone}})
    with torch.no_grad():
        outputs = mod({"inputs": {"input": image, "zone": test_zone}})
    for output in outputs[0]:
        print(output)
    mod.save(output_name)
    logger.info("generated torchscript model named {}".format(output_name))


if __name__ == "__main__":
    args_main(
        r'\\fs-ai\NCBiR_N\ComputerVision\object_detection\models\SoS2.6_NDS_extended2\yolox\SoS_2.6_NDS_extended2_s_416\yolox_s.py',
        'yoloxs', [], True, 1, r'C:\Users\arusak\export\yolox.torchscript.pt',
        r'\\fs-ai\NCBiR_N\ComputerVision\object_detection\models\SoS2.6_NDS_extended2\yolox\SoS_2.6_NDS_extended2_s_416\best_ckpt.pth')
