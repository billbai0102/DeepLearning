import torch
from torch import nn
from torchsummary import summary

from hyperparameters import *


class NLayerDiscriminator(nn.Module):
    def __init__(self):
        super(NLayerDiscriminator, self).__init__()
        layers = [
            nn.Conv2d(6, HIDDEN, 4, 2, 1)
            , nn.LeakyReLU(.2, inplace=True)
        ]

        cur_scale = 1
        prev_scale = 1

        for i in range(DIS_LAYERS):
            prev_scale = cur_scale
            cur_scale = 2 ** i
            layers += [
                nn.Conv2d(64 * prev_scale, 64 * cur_scale, 4, 2, 1, bias=False)
                , nn.BatchNorm2d(cur_scale * 64)
                , nn.LeakyReLU(.2, inplace=True)
            ]

        prev_scale = cur_scale

        layers += [
            nn.Conv2d(64 * prev_scale, 64 * 8, 4, 1, 1, bias=False)
            , nn.BatchNorm2d(64 * 8)
            , nn.LeakyReLU(.2, inplace=True),

            nn.Conv2d(64 * 8, 1, 4, 1, 1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # print(NLayerDiscriminator())
    print(summary(NLayerDiscriminator().cuda(), input_size=(6, 64, 64)))
