import torch
from torch import nn
from torchsummary import summary
import numpy as np

from hyperparameters import *


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1)
            , nn.Conv2d(channels, channels, 3)
            , nn.InstanceNorm2d(channels)
            , nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1)
            , nn.Conv2d(channels, channels, 3)
            , nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        """
        F(x) + x
        :param x: input
        :return: output of block
        """
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, res_blocks=9):
        super(Generator, self).__init__()

        def _layer(in_channels, out_channels, kernel_size, stride, padding, transpose=False):
            """
            create conv layer
            :param in_channels: in channels
            :param out_channels: out channels
            :param kernel_size: kernel size
            :param stride: conv stride
            :param padding: conv padding
            :param transpose: ConvTranspose2d or not
            :return: new block
            """
            layers = []

            if transpose:
                layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            return layers

        # input
        model = []
        model.append(nn.ReflectionPad2d(3))
        model += _layer(CHANNELS, GEN_HIDDEN, 7, 1, 0)

        # downsample
        model += _layer(GEN_HIDDEN * 1, GEN_HIDDEN * 2, 3, 2, 1)
        model += _layer(GEN_HIDDEN * 2, GEN_HIDDEN * 4, 3, 2, 1)

        # res blocks
        model += [ResidualBlock(GEN_HIDDEN * 4) for _ in range(res_blocks)]

        # upsample
        model += _layer(GEN_HIDDEN * 4, GEN_HIDDEN * 2, 3, 2, 1, transpose=True)
        model += _layer(GEN_HIDDEN * 2, GEN_HIDDEN * 1, 3, 2, 1, transpose=True)

        # output
        model += [
            nn.ReflectionPad2d(3)
            , nn.Conv2d(GEN_HIDDEN, CHANNELS, 7)
            , nn.Tanh()
        ]

        # build model
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    print(summary(Generator().cuda(), input_size=(3, 256, 256)))
    # print(Generator())
