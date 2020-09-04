import numpy as np
import torch
from torch import nn

from hyperparameters import *


class Discriminator(nn.Module):
    def __init__(self, classes, channels, img_sz, latent_dim):
        super(Discriminator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_sz = img_sz
        self.img_shape = (channels, img_sz, img_sz)
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.classes, self.classes)
        self.criterion = nn.BCELoss()

        def layer(in_sz, out_sz, dropout=True, activation=True):
            """
            Returns new fc layer w/ dropout and LeakyReLU
            :param in_sz: input size
            :param out_sz: output size
            :param dropout: dropout option
            :param activation: activation option
            :return: new fc layer
            """
            # fc layer
            layers = [nn.Linear(in_sz, out_sz)]
            # dropout layer
            if dropout:
                layers.append(nn.Dropout(.4))
            # activation function
            if activation:
                layers.append(nn.LeakyReLU(.2, inplace=True))

            return layers

        self.dis = nn.Sequential(
            *layer(self.classes + int(np.prod(self.img_shape)), DIS_HIDDEN * 8, dropout=False, activation=True)
            , *layer(DIS_HIDDEN * 8, DIS_HIDDEN * 4, dropout=True, activation=True)
            , *layer(DIS_HIDDEN * 4, DIS_HIDDEN * 2, dropout=False, activation=True)
            , *layer(DIS_HIDDEN * 2, DIS_HIDDEN * 1, dropout=True, activation=True)
            , *layer(DIS_HIDDEN * 1, 1, dropout=False, activation=False)

            , nn.Sigmoid()
        )

    def forward(self, img, labels):
        """
        Propagates through network
        :param img: (x) image
        :param labels: (y) labels
        :return: D(x|y)
        """
        img = img.view(img.size(0), -1)
        x = torch.cat((img, self.embedding(labels)), -1)
        return self.dis(x)

    def loss(self, output, label):
        return self.criterion(output, label)
