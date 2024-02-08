import unittest

import torch
from torch import nn

from yolox.models import IOUloss


class TestIOULoss(unittest.TestCase):
    def setUp(self):
        self.gt_bboxes = torch.tensor([[0.5, 0.5, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        self.perfect_bboxes = torch.tensor([[0.5, 0.5, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        self.bad_bboxes = torch.tensor([[1, 0.75, 1, 1], [1, 0.75, 1, 0.5], [0.75, 1, 0.5, 1], [1, 1, 0.5, 0.5]])

    def _test_loss(self, loss_function: nn.Module, gt_loss_values: torch.Tensor):
        loss_values = loss_function(self.perfect_bboxes, self.gt_bboxes)
        assert all([loss_value == 0. for loss_value in loss_values])
        loss_values = loss_function(self.bad_bboxes, self.gt_bboxes)
        assert torch.allclose(loss_values, gt_loss_values, atol=1e-4)

    def test_iuo_loss(self):
        self._test_loss(IOUloss(), torch.tensor([0.9467, 0.75, 0.75, 1.]))

    def test_giou_loss(self):
        self._test_loss(IOUloss(loss_type='giou'), torch.tensor([0.9026, 0.5, 0.5, 1.5]))
