import torch
from torch import nn
from torchsummary import summary

from hyperparameters import *


class Generator(nn.Module):
    def __init__(self, latent_dim=100, embed_in_dim=1024, embed_out_dim=128):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.embed_in_dim = embed_in_dim
        self.embed_out_dim = embed_out_dim

        self.embedding = nn.Sequential(
            nn.Linear(self.embed_in_dim, self.embed_out_dim)
            , nn.BatchNorm1d(self.embed_out_dim)
            , nn.LeakyReLU(.2, inplace=True)
        )

        model = []
        model += self._layer(self.latent_dim + self.embed_out_dim, 512, 4, 1, 0)
        model += self._layer(512, 256, 4, 2, 1)
        model += self._layer(256, 128, 4, 2, 1)
        model += self._layer(128, 64, 4, 2, 1)
        model += self._layer(64, CHANNELS, 4, 2, 1, out=True)

        self.model = nn.Sequential(
            *model
        )

    def forward(self, z, txt):
        txt = self.embedding(txt)
        txt = txt.view(txt.shape[0], txt.shape[1], 1, 1)
        z = torch.cat([txt, z], 1)
        return self.model(z)

    @staticmethod
    def _layer(in_channels, out_channels, kernel, stride, padding, out=False):
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, bias=(not out)))

        if out:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(.2, inplace=True))

        return layers


class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Embedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, out_channels)
            , nn.BatchNorm1d(out_channels)
            , nn.LeakyReLU(.2, inplace=True)
        )

    def forward(self, x, txt):
        out = self.embedding(txt)
        out = out.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        out = torch.cat([x, out], 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, embed_in_dim=1024, embed_out_dim=128):
        super(Discriminator, self).__init__()
        self.embed_in_dim = embed_in_dim
        self.embed_out_dim = embed_out_dim

        model = []
        model += self._layer(CHANNELS, 64, 4, 2, 1, False)
        model += self._layer(64, 128, 4, 2, 1)
        model += self._layer(128, 256, 4, 2, 1)
        model += self._layer(256, 512, 4, 2, 1)
        self.model = nn.Sequential(*model)

        self.embedding = Embedding(self.embed_in_dim, self.embed_out_dim)

        self.output = nn.Sequential(
            nn.Conv2d(512 + self.embed_out_dim, 1, 4, 1, 0, bias=False)
            , nn.Sigmoid()
        )

    def forward(self, x, txt):
        out_model = self.model(x)
        out = self.embedding(out_model, txt)
        out = self.output(out)

        return out.squeeze(), out_model

    def _layer(self, in_channels, out_channels, kernel, stride, padding, norm=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=(not norm)))

        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(.2, inplace=True))
        return layers


if __name__ == '__main__':
    pass
