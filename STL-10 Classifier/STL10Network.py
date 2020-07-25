"""
Neural Network class for the STL-10 classifeir
"""

import torch
from torch import nn
from torchsummary import summary
import yaml


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        with open('hyperparameters.yaml') as f:
            hp = yaml.safe_load(f)['hyperparameters']

        print('hi')

    @staticmethod
    def forward(self, x):
        return x


if __name__ == '__main__':
    net = NeuralNetwork().cuda()
    summary(net, input_size=(3, 96, 96))
