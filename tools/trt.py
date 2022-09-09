#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import shutil
from loguru import logger

import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch2trt import torch2trt

from functools import reduce
from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX ncnn deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("--fp16", default=False, action='store_true', help='enable fp16 optimization')
    parser.add_argument('--decode-in-inference', action='store_true', default=False, help='decode during inference. Must be set for inference on server')
    parser.add_argument("--use-onnx", default=False, action='store_true', help='Use intermediate onnx conversion')
    parser.add_argument("--opset", default=12, type=int, help="onnx opset version")
    parser.add_argument('--simplify', default=False, action='store_true', help='Simplify onnx model')
    parser.add_argument('--sparsify', default=False, action='store_true', help='sparsify model and apply TensorRT optimizations')
    parser.add_argument('--prop', default=0.3, type=float, help='sparsification amount')
    parser.add_argument('--struct', default=False, action='store_true', help="structured sparsification (not recommended)")
    parser.add_argument("--int8", default=False, action="store_true", help="enable int8 optimization")
    parser.add_argument("--calibrate", default=False, action='store_true', help="enable int8 calibration")
    parser.add_argument('--calib-num-images', type=int, default=200, help='number of images to be used for calibration')
    parser.add_argument("-cb", '--calib-batch-size', type=int, default=4, help='calibration batch size for INT8 optimization')
    parser.add_argument('--seed', default=10, type=int, help='set seed for INT8 calibration')
    parser.add_argument('--calib-algo', default='entropy2', choices=["minmax", "entropy", "entropy2"], help="calibration algorithm to be used for INT8 quantization")
    
    parser.add_argument(
        "-w", '--workspace', type=int, default=32, help='max workspace size in detect'
    )
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')
    parser.add_argument("--input", default=None, type=str, help="onnx input node name")
    parser.add_argument("--output", default=None, type=str, help="onnx output")
    return parser

def get_module_by_name(model, access_string):
    names = access_string.split('.')[:-1]
    return reduce(getattr, names, model)

@logger.catch
@torch.no_grad()
def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
        
    
    # INT8 calibration algorithm mappings
    ca_dict = {
        "entropy2": trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2,
        "entropy": trt.CalibrationAlgoType.ENTROPY_CALIBRATION,
        "minmax": trt.CalibrationAlgoType.MINMAX_CALIBRATION
    }

    model = exp.get_model()
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict

    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    model.eval()
    
    # model sparsification
    if args.sparsify:
        logger.info(f"sparsifying model")
        modules = [module[0] for module in model.named_parameters()]
        parameters_to_prune = []
        for module in modules:
            if 'weight' in module:
                obj = get_module_by_name(model, module)
                if args.struct and not isinstance(obj, nn.BatchNorm2d):
                    prune.ln_structured(obj, name="weight", amount=args.prop, n=1, dim=0)
                    prune.remove(obj, "weight")
                else:
                    parameters_to_prune.append((obj, "weight"))
        if not args.struct:
            parameters_to_prune = tuple(parameters_to_prune)
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=args.prop)
            for param in parameters_to_prune:
                prune.remove(*param)
        logger.info("model sparsification successful")
                    
                
    model.cuda()
    model.head.decode_in_inference = args.decode_in_inference
    x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()

    if args.fp16:
        logger.info("Running TensorRT with FP16 optimization")
        model_trt = torch2trt(
            model,
            [x],
            input_names=[args.input] if args.input is not None else None,
            output_names=[args.output] if args.ouput is not None else None,
            sparsify=args.sparsify,
            fp16_mode=True,
            log_level=trt.Logger.INFO,
            max_workspace_size=(1 << args.workspace),
            max_batch_size=args.batch,
	        use_onnx = args.use_onnx,
            opset=args.opset,
	        simplify = args.simplify
        )
    elif args.int8:
        logger.info("Running TensorRT with INT8 optimization")
        if args.calibrate:
            logger.info("Performing INT8 calibration")
            logger.info("Using calibration batch size of " +  str(args.calib_batch_size))
            logger.info("Calibrating with " + str(args.calib_num_images) + " images")
            from yolox.data import COCOCalibratorDataset, ValTransform
            dataset = COCOCalibratorDataset(
                data_dir = "../datasets/coco",
                name = "val2017",
                img_size = (exp.test_size[0], exp.test_size[1]),
                preproc = ValTransform(legacy=False),
                num_images = args.calib_num_images,
                seed=args.seed
            )
            model_trt = torch2trt(
                model,
                [x],
                input_names=[args.input] if args.input is not None else None,
                output_names=[args.output] if args.output is not None else None,
                sparsify=args.sparsify,
                int8_mode = True,
                int8_calib_dataset=dataset,
                int8_calib_batch_size= args.calib_batch_size,
                int8_calib_algorithm=ca_dict[args.calib_algo],
                log_level = trt.Logger.INFO,
                max_workspace_size = (1<<args.workspace),
                max_batch_size = args.batch,
		        use_onnx = args.use_onnx,
                opset = args.opset,
		        simplify = args.simplify
            )
        else:
            model_trt = torch2trt(
                model,
                [x],
                input_names=[args.input] if args.input is not None else None,
                output_names=[args.output] if args.output is not None else None,
                sparsify=args.sparsify,
                int8_mode = True,
                int8_calib_algorithm=ca_dict[args.calib_algo],
                log_level = trt.Logger.INFO,
                max_workspace_size = (1<<args.workspace),
                max_batch_size = args.batch,
		        use_onnx = args.use_onnx,
                opset = args.opset,
		        simplify = args.simplify
            )
    else:
        logger.info("Running TensorRT")
        model_trt = torch2trt(
            model,
            [x],
            input_names=[args.input] if args.input is not None else None,
            output_names=[args.output] if args.output is not None else None,
            sparsify=args.sparsify,
            log_level=trt.Logger.INFO,
            max_workspace_size=(1 << args.workspace),
            max_batch_size=args.batch,
	        use_onnx = args.use_onnx,
            opset=args.opset,
	        simplify = args.simplify
        )

    torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, "model_trt.engine")
    engine_file_demo = os.path.join("demo", "TensorRT", "cpp", "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    shutil.copyfile(engine_file, engine_file_demo)

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
