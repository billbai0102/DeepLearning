from torch import nn

from hyperparameters import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            # Block 1
            nn.Conv2d(CHANNELS, DIS_HIDDEN * 1, 4, 2, 1, bias=False)
            , nn.LeakyReLU(.2, inplace=True),

            # Block 2
            nn.Conv2d(DIS_HIDDEN * 1, DIS_HIDDEN * 2, 4, 2, 1, bias=False)
            , nn.BatchNorm2d(DIS_HIDDEN * 2)
            , nn.LeakyReLU(.2, inplace=True),

            # Block 3
            nn.Conv2d(DIS_HIDDEN * 2, DIS_HIDDEN * 4, 4, 2, 1, bias=False)
            , nn.BatchNorm2d(DIS_HIDDEN * 4)
            , nn.LeakyReLU(.2, inplace=True),

            # Block 4
            nn.Conv2d(DIS_HIDDEN * 4, DIS_HIDDEN * 8, 4, 2, 1, bias=False)
            , nn.BatchNorm2d(DIS_HIDDEN * 8)
            , nn.LeakyReLU(.2, inplace=True),

            # Output
            nn.Conv2d(DIS_HIDDEN * 8, CHANNELS, 4, 1, 0, bias=False)
            , nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        x = x.view(-1, 1)
        x = x.squeeze(1)
        return x

