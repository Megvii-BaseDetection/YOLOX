import argparse
import os

import cv2
import numpy as np
import torch

from loguru import logger
import time
import onnxruntime
import torchvision.transforms as transforms

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
from PIL import Image
import copy
np.set_printoptions(threshold=np.inf)

CLASSES = (
    'people','car'
)

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/whoami/Documents/Hanvon/yoloxs_0528.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="image",
        help="mode type, eg. image, video and webcam.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default='assets/xxx.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "--camid", 
        type=int, 
        default=0, 
        help="webcam demo camera id",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='outputs_imgs/total0614',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="320,320",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


def inference(args, origin_img):
    t0 = time.time()
    input_shape = tuple(map(int, args.input_shape.split(',')))

    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    
    predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

    
    return predictions, ratio

def image_process(args):
    folder_path = '/home/whoami/Documents/Hanvon/云台侦查/pics0607'

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        origin_img = cv2.imread(img_path)

        slice_size = (1080, 1080)

        count = 0
        total_time = 0
        copied_frame = origin_img.copy()

        # for y in [0, 300, 600]:
        #     for x in [0, 360, 720, 1080, 1440]:
        # for y in [0, 300, 600, 900, 1200, 1500, 1680]:
        #     for x in [0, 360, 720, 1080, 1440, 1800, 2160, 2520, 2880, 3240, 3360]:
        # for y in [0, 440]:
        #     for x in [0, 640, 1280]:

        for y in [0]:
            for x in [0, 840]:

                t0 = time.time()

                box = (x, y)
                print(f"box is {box}")
                
                slice_img = copied_frame[y:y + slice_size[1], x:x + slice_size[0]].copy()

                pred, ratio = inference(args, slice_img)
                print("ratio is ", ratio)
                boxes = pred[:, :4]
                scores = pred[:, 4:5] * pred[:, 5:]

                boxes_xyxy = np.ones_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
                boxes_xyxy /= ratio
                dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.5, score_thr=0.5)
                if dets is not None:
                    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                    slice_img = vis(slice_img, final_boxes, final_scores, final_cls_inds,
                                    conf=args.score_thr, class_names=CLASSES)
                    
                    # 写入原图
                    final_boxes[:, :4] += [box[0], box[1], box[0], box[1]]
                    origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                    conf=args.score_thr, class_names=CLASSES)


                logger.info("Infer time: {:.4f}s".format(time.time() - t0))
                total_time += time.time() - t0

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        output_path = os.path.join(args.output_path, filename)
        cv2.imwrite(output_path, origin_img)    
        print(f"Total used time is {total_time}s")


def imageflow_demo(args):
    cap = cv2.VideoCapture(args.input_path if args.mode == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    mkdir(args.output_path)
    current_time = time.localtime()
    save_folder = os.path.join(
        args.output_path, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.mode == "video":
        save_path = os.path.join(save_folder, args.input_path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")

    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            result_frame = inference(args, frame)
            vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

if __name__ == '__main__':
    args = make_parser().parse_args()
    if args.mode == "image":
        image_process(args)
    elif args.mode == "video" or args.mode == "webcam":
        imageflow_demo(args)


    