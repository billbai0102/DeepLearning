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
            , nn.LeakyReLU(0.2, inplace=True),

            # Block 2
            nn.ConvTranspose2d(GEN_HIDDEN * 8, GEN_HIDDEN * 4, 4, 2, 1, bias=False)
            , nn.BatchNorm2d(GEN_HIDDEN * 4)
            , nn.LeakyReLU(0.2, inplace=True),

            # Block 3
            nn.ConvTranspose2d(GEN_HIDDEN * 4, GEN_HIDDEN * 2, 4, 2, 1, bias=False)
            , nn.BatchNorm2d(GEN_HIDDEN * 2)
            , nn.LeakyReLU(0.2, inplace=True),

            # Block 4
            nn.ConvTranspose2d(GEN_HIDDEN * 2, GEN_HIDDEN * 1, 4, 2, 1, bias=False)
            , nn.BatchNorm2d(GEN_HIDDEN * 1)
            , nn.LeakyReLU(0.2, inplace=True),

            # Output block
            nn.ConvTranspose2d(GEN_HIDDEN * 1, CHANNELS, 4, 2, 1, bias=False)
            , nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x


if __name__ == '__main__':
    print(torchsummary.summary(Generator().cuda(), input_size=(100, 1, 1)))
