import torch
from torch import nn
import torchsummary

from hyperparameters import *


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Block 1
            nn.ConvTranspose2d(GEN_IN, GEN_HIDDEN * 8, 4, 1, 0, bias=False)
            , nn.BatchNorm2d(GEN_HIDDEN * 8)
            , nn.ReLU(inplace=True),

            # Block 2
            nn.ConvTranspose2d(GEN_IN, GEN_HIDDEN * 8, 4, 1, 0, bias=False)
            , nn.BatchNorm2d(GEN_HIDDEN * 8)
            , nn.ReLU(inplace=True),

            # Block 3
            nn.ConvTranspose2d(GEN_IN, GEN_HIDDEN * 4, 4, 1, 0, bias=False)
            , nn.BatchNorm2d(GEN_HIDDEN * 4)
            , nn.ReLU(inplace=True),

            # Block 4
            nn.ConvTranspose2d(GEN_IN, GEN_HIDDEN * 2, 4, 1, 0, bias=False)
            , nn.BatchNorm2d(GEN_HIDDEN * 2)
            , nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.gen(x)
