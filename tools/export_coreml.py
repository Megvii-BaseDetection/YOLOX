#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger
import sys
import torch
from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_modules
import coremltools as ct

def make_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="YOLOX ONNX Deployment")
    parser.add_argument("--output-name", type=str, default="yolox.onnx", help="Output name of models")
    parser.add_argument("--input", type=str, default="images", help="Input node name of ONNX model")
    parser.add_argument("--output", type=str, default="output", help="Output node name of ONNX model")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--dynamic", action="store_true", help="Whether the input shape should be dynamic or not")
    parser.add_argument("--no-onnxsim", action="store_true", help="Use onnxsim or not")
    parser.add_argument("--exp_file", type=str, default=None, help="Experiment description file")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--name", type=str, default=None, help="Model name")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    parser.add_argument("--decode_in_inference", action="store_true", help="Decode in inference or not")
    parser.add_argument("--class_name", type=str, default=None, help="Class name for the object")
  
    return parser


class YOLOXDetectModel(nn.Module):
    """Wrap an Ultralytics YOLO model for Apple iOS CoreML export."""

    def __init__(self, model, im, num_of_class, device='cuda'):
        """Initialize the YOLOXDetectModel class with a YOLO model and example image."""
        super().__init__()
        _, _, h, w = im.shape
        self.model = model
        self.nc = num_of_class  # number of classes
        self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h]).to(device)

    def forward(self, x):
        """Normalize predictions of object detection model with input size-dependent factors."""
        out_pred = self.model(x)
        xywh = out_pred[:, :, :4][0]
        objectness = out_pred[0][:, 4:5]
        class_conf = out_pred[0][:, 5:5 + self.nc]
        class_scores = objectness * class_conf
        return class_scores, xywh * self.normalize


@logger.catch
def main():
    """Main function to run the script."""
    args = make_parser().parse_args()
    logger.info(f"Arguments: {args}")

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    ckpt_file = args.ckpt if args.ckpt else os.path.join(exp.output_dir, args.experiment_name, "best_ckpt.pth")

    ckpt = torch.load(ckpt_file, map_location="cuda")
    if "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt)
    logger.info("Checkpoint loaded.")

    names = args.class_name # {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    nc = len(names.keys())  # number of classes

    im = torch.zeros(args.batch_size, 3, exp.test_size[0], exp.test_size[1]).to("cuda")
    model = YOLOXDetectModel(model, im, nc, "cuda")
    model.eval().to("cuda")

    y = model(im)
    traced_model = torch.jit.trace(model, im, strict=False)

    

    bias = [0.0, 0.0, 0.0]
    ct_model = ct.convert(
        traced_model,
        convert_to="neuralnetwork",
        inputs=[ct.ImageType("image", shape=im.shape, bias=bias)]
    )
    ct_model.save(args.output_name)

    weights_dir = args.output_name + "/Data/com.apple.CoreML/weights"
    _, _, h, w = list(im.shape)
    spec = ct_model.get_spec()
    out0, out1 = iter(spec.description.output)

    out0_shape = tuple(y[0].size())
    out1_shape = tuple(y[1].size())


    out0.type.multiArrayType.shape[:] = out0_shape
    out1.type.multiArrayType.shape[:] = out1_shape

    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    for i in range(2):
        decoder_output = _model._spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)

    nms_spec.description.output[0].name = "confidence"
    nms_spec.description.output[1].name = "coordinates"

    output_sizes = [nc, 4]
    for i in range(2):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
        ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
        del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = out0.name
    nms.coordinatesInputFeatureName = out1.name
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = 0.5
    nms.confidenceThreshold = 0.4
    nms.pickTop.perClass = True
    nms.stringClassLabels.vector.extend(names.values())
    nms_model = ct.models.MLModel(nms_spec)

    pipeline = ct.models.pipeline.Pipeline(
        input_features=[
            ("image", ct.models.datatypes.Array(3, h, w)),
            ("iouThreshold", ct.models.datatypes.Double()),
            ("confidenceThreshold", ct.models.datatypes.Double()),
        ],
        output_features=["confidence", "coordinates"],
    )

    # Model from spec
    _model = ct.models.MLModel(spec, weights_dir=weights_dir)
    
    pipeline.add_model(_model)
    pipeline.add_model(nms_model)

    pipeline.spec.description.input[0].ParseFromString(_model._spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

    pipeline.spec.specificationVersion = 5
    pipeline.spec.description.metadata.userDefined.update(
        {"IoU threshold": str(nms.iouThreshold), "Confidence threshold": str(nms.confidenceThreshold)}
    )

    ct_model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)
    ct_model.input_description["image"] = "Input image"
    ct_model.input_description["iouThreshold"] = f"(optional) IoU threshold override (default: {nms.iouThreshold})"
    ct_model.input_description["confidenceThreshold"] = (
        f"(optional) Confidence threshold override (default: {nms.confidenceThreshold})"
    )
    ct_model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
    ct_model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"

    metadata = {
        "description": "YOLOX",
        "author": "usesrname"
    }
    ct_model.short_description = metadata.pop("description")
    ct_model.author = metadata.pop("author")
    ct_model.user_defined_metadata.update({k: str(v) for k, v in metadata.items()})

    ct_model.save(args.output_name)


if __name__ == "__main__":
    main()
