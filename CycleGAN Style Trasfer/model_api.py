import torch

from hyperparameters import *
from base_model import BaseModel
from models.ResNetGenerator import Generator
from models.Discriminator import Discriminator


class GAN(BaseModel):
    def __init__(self, dl, test_dl):
        super(GAN, self).__init__()

        self.G_AB = Generator().cuda()
        self.G_BA = Generator().cuda()
        self.D_AB = Discriminator().cuda()
        self.D_BA = Discriminator().cuda()



