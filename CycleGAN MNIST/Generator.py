import numpy as np
import torch
from torch import nn

from hyperparameters import *


class Generator(nn.Module):
    def __init__(self, classes, channels, img_sz, latent_dim):
        super(Generator, self).__init__()

        self.classes = classes
        self.channels = channels
        self.img_sz = img_sz
        self.img_shape = (self.channels, self.img_sz, self.img_sz)
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.classes, self.classes)

        def layer(in_sz, out_sz, normalize=True):
            """
            Returns an fc layer w/ BatchNorm and LeakyReLU
            :param in_sz: input size
            :param out_sz: output size
            :param normalize: bool whether to apply BatchNorm
            :return: layer
            """
            # add fc layer (bias is False if BatchNorm comes after)
            layers = [nn.Linear(in_sz, out_sz, bias=(not normalize))]
            # add BatchNorm layer if normalize is True
            if normalize:
                layers.append(nn.BatchNorm1d(out_sz))
            # add LeakyReLU activation
            layers.append(nn.LeakyReLU(.2, inplace=True))

            return layers

        self.gen = nn.Sequential(
            *layer(self.latent_dim + self.classes, GEN_HIDDEN, False)
            , *layer(GEN_HIDDEN, GEN_HIDDEN * 2)
            , *layer(GEN_HIDDEN * 2, GEN_HIDDEN * 4)
            , *layer(GEN_HIDDEN * 4, GEN_HIDDEN * 8)

            , nn.Linear(GEN_HIDDEN * 8, int(np.prod(self.img_shape)))
            , nn.Tanh()
        )

    def forward(self, noise, labels):
        """
        Propagates through network
        :param noise: (z), noise to generate image
        :param labels: (y), label for cgan
        :return: G(z|y)
        """
        # noise
        z = torch.cat((self.embedding(labels), noise), -1)
        x = self.gen(z)

        # reshape to image
        x = x.view(x.size(0), *self.img_shape)
        return x
