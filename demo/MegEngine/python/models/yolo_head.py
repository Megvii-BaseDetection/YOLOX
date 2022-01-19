#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import megengine.functional as F
import megengine.module as M

from .network_blocks import BaseConv, DWConv


def meshgrid(x, y):
    """meshgrid wrapper for megengine"""
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    mesh_shape = (y.shape[0], x.shape[0])
    mesh_x = F.broadcast_to(x, mesh_shape)
    mesh_y = F.broadcast_to(y.reshape(-1, 1), mesh_shape)
    return mesh_x, mesh_y


class YOLOXHead(M.Module):
    def __init__(
        self, num_classes, width=1.0, strides=[8, 16, 32],
        in_channels=[256, 512, 1024], act="silu", depthwise=False
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # save for matching

        self.cls_convs = []
        self.reg_convs = []
        self.cls_preds = []
        self.reg_preds = []
        self.obj_preds = []
        self.stems = []
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                M.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                M.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                M.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                M.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                M.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.strides = strides
        self.grids = [F.zeros(1)] * len(in_channels)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        assert not self.training

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            output = F.concat([reg_output, F.sigmoid(obj_output), F.sigmoid(cls_output)], 1)
            outputs.append(output)

        self.hw = [x.shape[-2:] for x in outputs]
        # [batch, n_anchors_all, 85]
        outputs = F.concat([F.flatten(x, start_axis=2) for x in outputs], axis=2)
        outputs = F.transpose(outputs, (0, 2, 1))
        if self.decode_in_inference:
            return self.decode_outputs(outputs)
        else:
            return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([F.arange(hsize), F.arange(wsize)])
            grid = F.stack((xv, yv), 2).reshape(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = (
            output.permute(0, 1, 3, 4, 2)
            .reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = F.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            xv, yv = meshgrid(F.arange(hsize), F.arange(wsize))
            grid = F.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(F.full((*shape, 1), stride))

        grids = F.concat(grids, axis=1)
        strides = F.concat(strides, axis=1)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = F.exp(outputs[..., 2:4]) * strides
        return outputs
