from torch import nn

import time
import os


class BaseModel:
    def __init__(self):
        if os.path.exists('./progress_images'):
            os.mkdir('./progress_images')
        if os.path.exists('./pt'):
            os.mkdir('./pt')

    @property
    def generator(self):
        """
        returns generator
        """
        pass

    @property
    def discriminator(self):
        """
        Returns discriminator
        """
        pass

    def train(self):
        """
        train loop
        """
        pass

    @classmethod
    def init_weights(cls, layer):
        """
        Initializes weights for a layer
        :param layer: layer
        """
        layer_name = layer.__class__.__name__

        if layer_name.find('Conv') != -1:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)

        if layer_name.find('BatchNorm') != -1:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0.0)
