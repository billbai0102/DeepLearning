import torch
from torch import nn
from torchsummary import summary

from Hyperparameters import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def _layer(in_channels, out_channels, stride=1, padding=1, norm=True):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=(not norm)))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(.2))
            return layers

        self.discriminator = nn.Sequential(
            *_layer(3, HIDDEN, norm=False)
            , *_layer(HIDDEN, HIDDEN, stride=2)
            , *_layer(HIDDEN, HIDDEN * 2)
            , *_layer(HIDDEN * 2, HIDDEN * 2, stride=2)
            , *_layer(HIDDEN * 2, HIDDEN * 4)
            , *_layer(HIDDEN * 4, HIDDEN * 4, stride=2)
            , *_layer(HIDDEN * 4, HIDDEN * 8)
            , *_layer(HIDDEN * 8, HIDDEN * 8, stride=2),

            nn.AdaptiveAvgPool2d(1)
            , nn.Conv2d(HIDDEN * 8, HIDDEN * 16, 1)
            , nn.LeakyReLU(.2)
            , nn.Conv2d(HIDDEN * 16, 1, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.discriminator(x).view(x.size(0)))
