#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import preproc, sliding_window
from yolox.data.datasets import COCO_CLASSES, VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    """
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    """
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        #fp16=False,
        #legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        #self.fp16 = fp16
        #self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)


    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        
        #img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=(0,0,0))
        #ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        #img_info["ratio"] = ratio
        """
        if (img.shape[0]>exp.test_size[0]):
            h_r = (img.shape[0]//exp.test_size[0]+1)*exp.test_size[0]-img.shape[0]
        elif(img.shape[0]<exp.test_size[0]):
            h_r = (exp.test_size[0]-img.shape[0])
        else:
            h_r = 0
        if (img.shape[1]>exp.test_size[1]):
            w_r = (img.shape[1]//exp.test_size[1]+1)*exp.test_size[1]-img.shape[1]
        elif(img.shape[1]<exp.test_size[1]):
            w_r = (exp.test_size[1]-img.shape[1])
        else:
            w_r = 0
        top = h_r//2
        bottom =  h_r-top
        left = w_r//2
        right =  w_r-left
        print("top: ",top)
        print("left: ",left)
        print("original img size: ",img.shape)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=(0,0,0))              
        print("after img size: ",img.shape)
        """
        (winW, winH) = (exp.test_size[1], exp.test_size[0])
        (imgW, imgH)= (img.shape[1],img.shape[0])
        if (imgH%winH):
            y_stepSize = winH-(winH*(imgH//winH+1)-imgH)//(imgH//winH)
            if(imgW%winW):
                x_stepSize = winW-(winW*(imgW//winW+1)-imgW)//(imgW//winW)
            else:
                x_stepSize = winW
        else:
            y_stepSize = winH
            if(imgW%winW):
                x_stepSize = winW-(winW*(imgW//winW+1)-imgW)//(imgW//winW)
            else:
                x_stepSize = winW
        numW = 0
        for (x, y, window) in sliding_window(img, YstepSize=y_stepSize, XstepSize=x_stepSize, windowSize=(winW, winH)):
		    # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            Wimg, _ = preproc(window, self.test_size, self.rgb_means, self.std)
            Wimg = torch.from_numpy(Wimg).unsqueeze(0)
            Wimg = Wimg.float()
            if self.device == "gpu":
                Wimg = Wimg.cuda()
                #if self.fp16:
                #    Wimg = Wimg.half()  # to FP16

            with torch.no_grad():
                t0 = time.time()
                Woutputs = self.model(Wimg)
                if numW != 0:
                    Woutputs[:, :, 0] = torch.add(Woutputs[:, :,0], x)
                    Woutputs[:, :, 1] = torch.add(Woutputs[:, :,1], y)
                    outputs = torch.cat((outputs, Woutputs), 1) 
                else:
                    outputs = Woutputs
                    numW=numW+1

        if self.decoder is not None:
            outputs = self.decoder(outputs, dtype=outputs.type())
        outputs = postprocess(
            outputs, self.num_classes, self.confthre,
            self.nmsthre#, class_agnostic=True
        )
        """
        if outputs[0] is None:
            pass
        elif len(outputs[0]) == 2:
            li_outputs = []
            temp = torch.empty(1, 7)
            temp[0][0] = torch.min(outputs[0][0, 0], outputs[0][1, 0])
            temp[0][1] = torch.min(outputs[0][0, 1], outputs[0][1, 1])
            temp[0][2] = torch.max(outputs[0][0, 2], outputs[0][1, 2])
            temp[0][3] = torch.max(outputs[0][0, 3], outputs[0][1, 3])
            temp[0][4] = torch.add(outputs[0][0, 4], outputs[0][1, 4]) / 2
            temp[0][5] = torch.add(outputs[0][0, 5], outputs[0][1, 5]) / 2
            temp[0][6] = torch.add(outputs[0][0, 6], outputs[0][1, 6]) / 2
            li_outputs.append(temp)
            outputs = li_outputs

        """
        logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        #ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            class_count = {}
            class_AP = {}
            for i in self.cls_names:
                class_count[i] = 0
                class_AP[i] = 0.0
            x0 = 15
            y0 = 0
            row = 0
            for k in class_count: 
                if((y0+row+50)>=img.shape[0]):
                    x0 = x0+200
                    y0 = 25
                    row = 0
                else:
                    row = row+25
                cv2.putText(img, str(k)+": "+str(class_count[k]), (x0,y0+row), font, 0.8, (0, 255, 255), thickness=2)
                if class_count[k] !=0:
                    class_AP[k]=class_AP[k]/class_count[k]
                else:
                    class_AP[k]=0.0
                row = row+25
                cv2.putText(img, "AP"+": "+'{:.1f}%'.format(class_AP[k]), (x0,y0+row), font, 0.8, (0, 255, 255), thickness=2)
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        #bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device#, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
