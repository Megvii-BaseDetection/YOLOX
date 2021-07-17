#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# NOTE: this file is not finished.
import sys
import tempfile
import time
from tqdm import tqdm

import torch

from yolox.data.dataset.vocdataset import ValTransform
from yolox.utils import get_rank, is_main_process, make_pred_vis, make_vis, synchronize


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = dist.scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        print("num_imgs: ", len(image_ids))
        print("last img_id: ", image_ids[-1])
        print(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


class VOCEvaluator:
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """

    def __init__(self, data_dir, img_size, confthre, nmsthre, vis=False):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        test_sets = [("2007", "test")]
        self.dataset = VOCDetection(
            root=data_dir,
            image_sets=test_sets,
            input_dim=img_size,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        )
        self.num_images = len(self.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0
        )
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.vis = vis

    def evaluate(self, model, distributed=False):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        model.eval()
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        ids = []
        data_dict = []
        dataiterator = iter(self.dataloader)
        img_num = 0
        indices = list(range(self.num_images))
        dis_indices = indices[get_rank() :: distributed_util.get_world_size()]
        progress_bar = tqdm if distributed_util.is_main_process() else iter
        num_classes = 20
        predictions = {}

        if is_main_process():
            inference_time = 0
            nms_time = 0
            n_samples = len(dis_indices)

        for i in progress_bar(dis_indices):
            img, _, info_img, id_ = self.dataset[i]  # load a batch
            info_img = [float(info) for info in info_img]
            ids.append(id_)
            with torch.no_grad():
                img = Variable(img.type(Tensor).unsqueeze(0))

                if is_main_process() and i > 9:
                    start = time.time()

                if self.vis:
                    outputs, fuse_weights, fused_f = model(img)
                else:
                    outputs = model(img)

                if is_main_process() and i > 9:
                    infer_end = time.time()
                    inference_time += infer_end - start

                outputs = postprocess(outputs, 20, self.confthre, self.nmsthre)

                if is_main_process() and i > 9:
                    nms_end = time.time()
                    nms_time += nms_end - infer_end

                if outputs[0] is None:
                    predictions[i] = (None, None, None)
                    continue
                outputs = outputs[0].cpu().data

            bboxes = outputs[:, 0:4]
            bboxes[:, 0::2] *= info_img[0] / self.img_size[0]
            bboxes[:, 1::2] *= info_img[1] / self.img_size[1]
            cls = outputs[:, 6]
            scores = outputs[:, 4] * outputs[:, 5]
            predictions[i] = (bboxes, cls, scores)

            if self.vis:
                o_img, _, _, _ = self.dataset.pull_item(i)
                make_vis("VOC", i, o_img, fuse_weights, fused_f)
                class_names = self.dataset._classes

                bbox = bboxes.clone()
                bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
                bbox[:, 3] = bbox[:, 3] - bbox[:, 1]

                make_pred_vis("VOC", i, o_img, class_names, bbox, cls, scores)

            if is_main_process():
                o_img, _, _, _ = self.dataset.pull_item(i)
                class_names = self.dataset._classes
                bbox = bboxes.clone()
                bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
                bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
                make_pred_vis("VOC", i, o_img, class_names, bbox, cls, scores)

        synchronize()
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        if not is_main_process():
            return 0, 0

        print("Main process Evaluating...")

        a_infer_time = 1000 * inference_time / (n_samples - 10)
        a_nms_time = 1000 * nms_time / (n_samples - 10)

        print(
            "Average forward time: %.2f ms, Average NMS time: %.2f ms, Average inference time: %.2f ms"
            % (a_infer_time, a_nms_time, (a_infer_time + a_nms_time))
        )

        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(num_classes)]
        for img_num in range(self.num_images):
            bboxes, cls, scores = predictions[img_num]
            if bboxes is None:
                for j in range(num_classes):
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                continue
            for j in range(num_classes):
                mask_c = cls == j
                if sum(mask_c) == 0:
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                    continue

                c_dets = torch.cat((bboxes, scores.unsqueeze(1)), dim=1)
                all_boxes[j][img_num] = c_dets[mask_c].numpy()

            sys.stdout.write(
                "im_eval: {:d}/{:d} \r".format(img_num + 1, self.num_images)
            )
            sys.stdout.flush()

        with tempfile.TemporaryDirectory() as tempdir:
            mAP50, mAP70 = self.dataset.evaluate_detections(all_boxes, tempdir)
            return mAP50, mAP70
