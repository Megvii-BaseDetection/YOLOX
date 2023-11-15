#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Copyright (c) Francisco Massa.
# Copyright (c) Ellis Brown, Max deGroot.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import os.path
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import CacheDataset, cache_read_img
from .voc_classes import VOC_CLASSES


class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES)))
        )
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)

        return res, img_info


class VOCDetection(CacheDataset):

    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
        self,
        data_dir,
        image_sets=[("2007", "trainval"), ("2012", "trainval")],
        img_size=(416, 416),
        preproc=None,
        target_transform=AnnotationTransform(),
        dataset_name="VOC0712",
        cache=False,
        cache_type="ram",
    ):
        self.root = data_dir
        self.image_set = image_sets
        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self._classes = VOC_CLASSES
        self.cats = [
            {"id": idx, "name": val} for idx, val in enumerate(VOC_CLASSES)
        ]
        self.class_ids = list(range(len(VOC_CLASSES)))
        self.ids = list()
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, "VOC" + year)
            for line in open(
                os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
            ):
                self.ids.append((rootpath, line.strip()))
        self.num_imgs = len(self.ids)

        self.annotations = self._load_coco_annotations()

        path_filename = [
            (self._imgpath % self.ids[i]).split(self.root + "/")[1]
            for i in range(self.num_imgs)
        ]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=self.root,
            cache_dir_name=f"cache_{self.name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(self.num_imgs)]

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()

        assert self.target_transform is not None
        res, img_info = self.target_transform(target)
        height, width = img_info

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img

    def load_image(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        assert img is not None, f"file named {self._imgpath % img_id} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        target, img_info, _ = self.annotations[index]
        img = self.read_img(index)

        return img, target, img_info, index

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = os.path.join(self.root, "VOC" + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        cachedir = os.path.join(
            self.root, "annotations_cache", "VOC" + self._year, name
        )
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(VOC_CLASSES):

            if cls == "__background__":
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return np.mean(aps)
