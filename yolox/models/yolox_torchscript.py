"""
Deployment wrapper for export YoloX model to TorchScript. Required by task DSVA-1888.
"""

import torch
import torch.nn as nn


class DeployModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, targets=None):
        image = x.permute(0, 3, 1, 2)
        return self.model(image)