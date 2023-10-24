#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# VOC_CLASSES = ( '__background__', # always index 0
VOC_CLASSES = (
    "drone",
    "cat",
)

# def loadClassesAsTuple():
#     classes = []
#     with open('../../../datasets/VOCdevkit/VOC2012/labelmap.txt', 'r') as file:
#         for line in file.readlines():
#             classes.append(line.replace(" ", "").replace("\n", ""))
#
#     return tuple(classes)


# VOC_CLASSES = loadClassesAsTuple()

# if __name__ == '__main__':
#     c = loadClassesAsTuple()
