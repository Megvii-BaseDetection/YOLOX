#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import unittest

import torch
from torch import nn

from yolox.utils import adjust_status
from yolox.exp import get_exp


class TestModelUtils(unittest.TestCase):

    def setUp(self):
        self.model: nn.Module = get_exp(exp_name="yolox-s").get_model()

    def test_model_state_adjust_status(self):
        data = torch.ones(1, 10, 800, 800)
        # use bn since bn changes state during train/val
        model = nn.BatchNorm2d(10)
        prev_state = model.state_dict()

        modes = [False, True]
        results = [True, False]

        # test under train/eval mode
        for mode, result in zip(modes, results):
            with adjust_status(model, training=mode):
                model(data)
            model_state = model.state_dict()
            self.assertTrue(len(model_state) == len(prev_state))
            self.assertEqual(
                result,
                all([torch.allclose(v, model_state[k]) for k, v in prev_state.items()])
            )

        # test recurrsive context case
        prev_state = model.state_dict()
        with adjust_status(model, training=False):
            with adjust_status(model, training=False):
                model(data)
        model_state = model.state_dict()
        self.assertTrue(len(model_state) == len(prev_state))
        self.assertTrue(
            all([torch.allclose(v, model_state[k]) for k, v in prev_state.items()])
        )

    def test_model_effect_adjust_status(self):
        # test context effect
        self.model.train()
        with adjust_status(self.model, training=False):
            for module in self.model.modules():
                self.assertFalse(module.training)
        # all training after exit
        for module in self.model.modules():
            self.assertTrue(module.training)

        # only backbone set to eval
        self.model.backbone.eval()
        with adjust_status(self.model, training=False):
            for module in self.model.modules():
                self.assertFalse(module.training)

        for name, module in self.model.named_modules():
            if "backbone" in name:
                self.assertFalse(module.training)
            else:
                self.assertTrue(module.training)


if __name__ == "__main__":
    unittest.main()
