import torch
from torch import nn
from torchsummary import summary

from Hyperparameters import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
