#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
from pathlib import Path

from yolox.exp import CustomExp


class Exp(CustomExp):
    def __init__(self):
        """
        这里只罗列出部分参数，详细参数设置请参见CustomExp
        """
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # 类别数
        self.num_classes = 2

        # train和val json路径
        self.data_dir = "datasets/CustomDataset"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        # 训练代数
        self.max_epoch = 300

        self.data_num_workers = 4
