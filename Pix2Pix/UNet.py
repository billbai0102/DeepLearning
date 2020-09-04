import torch
from torch import nn

from hyperparameters import *


class UNetSkipBlock(nn.Module):
    def __init__(self, out_channels, in_channels, submodule, dropout):
        super(UNetSkipBlock, self).__init__()
        down = [
            nn.LeakyReLU(.2, inplace=True)
            , nn.Conv2d(out_channels, in_channels, 4, 2, 1, bias=False)
            , nn.BatchNorm2d(in_channels)
        ]

        up = [
            nn.LeakyReLU(.2, inplace=True)
            , nn.ConvTranspose2d(in_channels * 2, out_channels, 4, 2, 1, bias=False)
            , nn.BatchNorm2d(out_channels)
        ]

        # apply dropout
        if dropout:
            block = down + [submodule] + up + [nn.Dropout(0.4)]
        else:
            block = down + [submodule] + up
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class UNetSkipInnermostBlock(nn.Module):
    def __init__(self, out_channels, in_channels):
        super(UNetSkipInnermostBlock, self).__init__()
        down = [
            nn.LeakyReLU(.2, inplace=True)
            , nn.Conv2d(out_channels, in_channels, 4, 2, 1, bias=False)
        ]

        up = [
            nn.LeakyReLU(.2, inplace=True)
            , nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            , nn.BatchNorm2d(out_channels)
        ]

        self.block = nn.Sequential(*(down + up))

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class UNetSkipOutermostBlock(nn.Module):
    def __init__(self, out_channels, in_channels, submodule):
        super(UNetSkipOutermostBlock, self).__init__()
        down = [
            nn.Conv2d(out_channels, in_channels, 4, 2, 1, bias=False)
        ]

        up = [
            nn.LeakyReLU(.2, inplace=True)
            , nn.ConvTranspose2d(in_channels * 2, out_channels, 4, 2, 1)
            , nn.Tanh()
        ]

        self.block = nn.Sequential(*(down + [submodule] + up))

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    '''
    \______________ |   <- outermost
     \_____________|
      \___________|
       \_________|
        \_______|
         \_____|
          \___|
           \_|          <- innermost
    '''
    def __init__(self):
        super(UNet, self).__init__()

        unet_block = UNetSkipInnermostBlock(HIDDEN * 8, HIDDEN * 8)
        for i in range(3):
            unet_block = UNetSkipBlock(HIDDEN * 8, HIDDEN * 8, submodule=unet_block, dropout=True)
        unet_block = UNetSkipBlock(HIDDEN * 4, HIDDEN * 8, submodule=unet_block, dropout=False)
        unet_block = UNetSkipBlock(HIDDEN * 2, HIDDEN * 4, submodule=unet_block, dropout=False)
        unet_block = UNetSkipBlock(HIDDEN * 1, HIDDEN * 2, submodule=unet_block, dropout=True)
        self.unet = UNetSkipOutermostBlock(3, HIDDEN * 1, submodule=unet_block)

    def forward(self, x):
        return self.unet(x)


if __name__ == '__main__':
    print(UNet())
