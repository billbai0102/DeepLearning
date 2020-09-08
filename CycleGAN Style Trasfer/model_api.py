import torch
from torch import nn
from torch import optim

import time
import random
import itertools
import numpy as np

from hyperparameters import *
from base_model import BaseModel
from models.ResNetGenerator import Generator
from models.Discriminator import Discriminator


class Buffer:
    def __init__(self, depth):
        """
        Buffer class - Allows Discriminators to sample images
        :param depth:
        """
        self.depth = depth
        self.buffer = []

    def update(self, image):
        if len(self.buffer) == self.depth:
            idx = random.randint(0, self.depth-1)
            self.buffer[idx] = image
        else:
            self.buffer.append(image)

        if random.uniform(0, 1) > .5:
            idx = random.randint(0, len(self.buffer) - 1)
            return self.buffer[idx]
        else:
            return image


class GAN(BaseModel):
    def __init__(self, dl, test_dl):
        super(GAN, self).__init__()

        self.dl = dl
        self.test_dl = test_dl

        # instantiate generators and initialize weights
        self.G_AB = Generator().cuda()
        self.G_BA = Generator().cuda()
        self.G_AB.apply(GAN.init_weights)
        self.G_BA.apply(GAN.init_weights)

        # instantiate discriminators and initialize weights
        self.D_A = Discriminator().cuda()
        self.D_B = Discriminator().cuda()
        self.D_A.apply(GAN.init_weights)
        self.D_B.apply(GAN.init_weights)

        # optimizers
        self.optim_G = None
        self.optim_D_A = None
        self.optim_D_B = None
        self._init_optim()

        # loss functions
        self.crit_adv = nn.MSELoss()  # adversarial loss
        self.crit_cyc = nn.L1Loss()   # cycle loss
        self.crit_ide = nn.L1Loss()   # identity loss

    @property
    def generator_AB(self):
        return self.G_AB

    @property
    def generator_BA(self):
        return self.G_BA

    @property
    def discriminator_A(self):
        return self.D_A

    @property
    def discriminator_B(self):
        return self.D_B

    def _init_optim(self):
        self.optim_G = optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters())
            , lr=LR
            , betas=BETAS)

        self.optim_D_A = optim.Adam(
            filter(
                lambda param: param.requires_grad
                , self.D_A.parameters()
            )
            , lr=LR
            , betas=BETAS
        )

        self.optim_D_B = optim.Adam(
            filter(
                lambda param: param.requires_grad
                , self.D_B.parameters()
            )
            , lr=LR
            , betas=BETAS
        )

    def train(self, epochs):
        """
        training loop
        :param epochs: epochs to train for
        """
        # set train flag for all models
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

        # tensors for real and fake data
        real_label = torch.ones((self.dl.batch_size, 1, IMG_SIZE//2**4, IMG_SIZE//2**4)).cuda()
        fake_label = torch.zeros((self.dl.batch_size, 1, IMG_SIZE // 2 ** 4, IMG_SIZE // 2 ** 4)).cuda()

        # buffers
        buffer_A = Buffer(50)
        buffer_B = Buffer(50)

        # train loop
        for epoch in range(epochs):
            running_time = time.time()
            for idx, data in enumerate(self.dl):
                # move data to CUDA
                real_A = data['train_A'].cuda()
                real_B = data['train_B'].cuda()

                '''Train Generators'''
                # adversarial loss
                self.optim_G.zero_grad()
                fake_B = self.G_AB(real_A)
                fake_A = self.G_BA(real_B)
                loss_adv_B = self.crit_adv(self.D_B(fake_B), real_label)
                loss_adv_A = self.crit_adv(self.D_A(fake_A), real_label)
                loss_adv = (loss_adv_B + loss_adv_A) / 2

                # cycle loss
                cyc_B = self.G_AB(fake_A)
                cyc_A = self.G_BA(fake_B)
                loss_cyc_B = self.crit_cyc(cyc_B, real_B)
                loss_cyc_A = self.crit_cyc(cyc_A, real_A)
                loss_cyc = (loss_cyc_B + loss_cyc_A) / 2

                # identity loss
                loss_ide_B = self.crit_ide(self.G_AB(real_B), real_B)
                loss_ide_A = self.crit_ide(self.G_BA(real_A), real_A)
                loss_ide = (loss_ide_B + loss_ide_A) / 2

                loss_G = loss_adv + (10 * loss_cyc) + (5 * loss_ide)
                loss_G.backward()
                self.optim_G.step()

                '''Train Discriminators'''
                self.optim_D_A.zero_grad()
                self.optim_D_B.zero_grad()

                











