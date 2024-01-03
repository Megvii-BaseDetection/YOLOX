import torch
import torch.nn as nn
from yolox.models.yolox import YOLOX


class DeployModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.yolox = YOLOX()

    def forward(self, x: dict[str, dict[str, torch.Tensor]], targets=None):
        image = x['inputs']['input'].permute(0, 3, 1, 2)
        return self.yolox(image)