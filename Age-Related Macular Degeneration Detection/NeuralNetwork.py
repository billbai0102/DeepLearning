import torch
from torch import nn
from torchsummary import summary
import yaml


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    @staticmethod
    def forward(x):
        return x

