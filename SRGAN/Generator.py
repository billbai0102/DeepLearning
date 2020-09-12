import torch
from torch import nn
from torchsummary import summary

import math

from Hyperparameters import *


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
            , nn.BatchNorm2d(channels)
            , nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
            , nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels, scale):
        super(UpSampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * (scale ** 2), 3, 1, 1)
            , nn.PixelShuffle(scale)
            , nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, scale):
        super(Generator, self).__init__()
        upsample_blocks = int(math.log(scale, 2))

        # input block
        self.block1_I = nn.Sequential(
            nn.Conv2d(3, HIDDEN, 9, 1, 4)
            , nn.PReLU()
        )

        # residual blocks
        self.block2_R = ResidualBlock(HIDDEN)
        self.block3_R = ResidualBlock(HIDDEN)
        self.block4_R = ResidualBlock(HIDDEN)
        self.block5_R = ResidualBlock(HIDDEN)
        self.block6_R = ResidualBlock(HIDDEN)

        # transition block
        self.block7_T = nn.Sequential(
            nn.Conv2d(HIDDEN, HIDDEN, 3, 1, 1, bias=False)
            , nn.BatchNorm2d(HIDDEN)
        )

        # upsample blocks
        self.block8_U = nn.Sequential(
            *[UpSampleBlock(HIDDEN, 2) for _ in range(upsample_blocks)]
            , nn.Conv2d(HIDDEN, 3, 9, 1, 4)
        )

    def forward(self, x):
        pass1 = self.block1_I(x)
        pass2 = self.block2_R(pass1)
        pass3 = self.block3_R(pass2)
        pass4 = self.block4_R(pass3)
        pass5 = self.block5_R(pass4)
        pass6 = self.block6_R(pass5)
        pass7 = self.block7_T(pass6)
        out = self.block8_U(pass1 + pass7)

        return (torch.tanh(out) + 1) / 2


if __name__ == '__main__':
    print(summary(Generator(4).cuda(), input_size=(3, 256, 256)))
