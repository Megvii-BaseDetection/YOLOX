import sys

sys.path.insert(0, '../../YOLOX')
import torch
import cv2
from loguru import logger
from YOLOX.yolox.data.data_augment import preproc
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.utils import postprocess
from Fusion.utils.visualize import vis
import os
import time

COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            decoder=None,
            device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.rgb_means = COCO_MEAN
        self.std = COCO_STD

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

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res
