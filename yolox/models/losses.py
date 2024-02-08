#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
from loguru import logger
import torch
import torch.nn as nn


def get_smalest_enclosing_box(box1, box2) -> tuple[torch.Tensor, torch.Tensor]:
    box_top_left = torch.min(
        (box1[:, :2] - box1[:, 2:] / 2), (box2[:, :2] - box2[:, 2:] / 2)
    )
    box_bottom_right = torch.max(
        (box1[:, :2] + box1[:, 2:] / 2), (box2[:, :2] + box2[:, 2:] / 2)
    )
    return box_top_left, box_bottom_right


def get_d_c_parameters(box1, box2) -> tuple[torch.Tensor, torch.Tensor]:
    d = (box1[:, :2] - box2[:, :2]).pow(2).sum(1).sqrt()
    c_tl, c_br = get_smalest_enclosing_box(box1, box2)
    c = (c_br - c_tl).pow(2).sum(1).sqrt()
    return d, c


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def __repr__(self):
        return f"{self.__class__.__name__}(loss_type={self.loss_type})"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        # pred and target shape = [c_x, c_y, w, h]
        top_left_intersection = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        bottom_right_intersection = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_pred = torch.prod(pred[:, 2:], 1)
        area_target = torch.prod(target[:, 2:], 1)

        en = (top_left_intersection < bottom_right_intersection).type(top_left_intersection.type()).prod(dim=1)
        area_intersection = torch.prod(bottom_right_intersection - top_left_intersection, 1) * en
        area_union = area_pred + area_target - area_intersection
        iou = area_intersection / (area_union + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2

        elif self.loss_type == "giou":
            c_tl, c_br = get_smalest_enclosing_box(pred, target)
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_union) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
