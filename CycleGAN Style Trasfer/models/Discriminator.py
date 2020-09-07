import torch
from torch import nn
from torchsummary import summary

from hyperparameters import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def _layer(in_channels, out_channels, stride, normalize=True):
            """
            Creates a conv2d block
            :param in_channels: input channels
            :param out_channels: output channels
            :param stride: stride size
            :param normalize: option for instancenorm2d
            :return: new conv block
            """
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *_layer(CHANNELS, DIS_HIDDEN, 2, normalize=False)
            , *_layer(DIS_HIDDEN, DIS_HIDDEN * 2, 2, normalize=True)
            , *_layer(DIS_HIDDEN * 2, DIS_HIDDEN * 4, 2, normalize=True)
            , *_layer(DIS_HIDDEN * 4, DIS_HIDDEN * 8, 1, normalize=True)
            , nn.Conv2d(DIS_HIDDEN * 8, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)
