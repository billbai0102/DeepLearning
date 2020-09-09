import torch
from torch import nn
from torch import optim
from torchvision.utils import save_image

import os
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
        self.cycles = 0

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

        # hist
        self.hist = {
            'loss_D': [],
            'loss_G': [],
        }

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

    def hist(self):
        return self.hist

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

    def save_models(self, epoch):
        torch.save(self.G_AB.state_dict(), f'./pt/epoch{epoch}/G_AB.pt')
        torch.save(self.G_BA.state_dict(), f'./pt/epoch{epoch}/G_BA.pt')
        torch.save(self.D_A.state_dict(), f'./pt/epoch{epoch}/D_A.pt')
        torch.save(self.D_B.state_dict(), f'./pt/epoch{epoch}/D_B.pt')

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
        
        # IMG_SIZE//2**4
        # tensors for real and fake data
        real_label = torch.ones((self.dl.batch_size, 1, 29, 29)).cuda()
        fake_label = torch.zeros((self.dl.batch_size, 1, 29, 29)).cuda()

        # buffers
        buffer_A = Buffer(50)
        buffer_B = Buffer(50)

        # train loop
        for epoch in range(epochs):
            running_time = time.time()
            for idx, data in enumerate(self.dl):
                self.cycles += 1
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
#                 print(fake_B.shape)
#                 print(fake_A.shape)
#                 print(cyc_B.shape)
#                 print(cyc_A.shape)
#                 print(real_B.shape)
#                 print(real_A.shape)
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
                # train discriminator A
                self.optim_D_A.zero_grad()
                fake_A = buffer_A.update(fake_A)
                loss_real = self.crit_adv(self.D_A(real_A), real_label)
                loss_fake = self.crit_adv(self.D_A(fake_A.detach()), fake_label)
                loss_D_A = (loss_real + loss_fake) / 2
                loss_D_A.backward()
                self.optim_D_A.step()

                # train discriminator B
                self.optim_D_B.zero_grad()
                fake_B = buffer_B.update(fake_B)
                loss_real = self.crit_adv(self.D_B(real_B), real_label)
                loss_fake = self.crit_adv(self.D_B(fake_B.detach()), fake_label)
                loss_D_B = (loss_real + loss_fake) / 2
                loss_D_B.backward()
                self.optim_D_B.step()

                loss_D = (loss_D_A + loss_D_B) / 2

                '''Status updates'''
                if self.cycles % 100 == 0:
                    loss_D = loss_D.mean().item()
                    loss_G = loss_G.mean().item()

                    # add loss to history
                    self.hist['loss_D'].append(loss_D)
                    self.hist['loss_G'].append(loss_G)

                    # save sample images of progress
                    with torch.no_grad():
                        data = next(iter(self.test_dl))
                        real_A = data['test_A'].cuda()
                        real_B = data['test_B'].cuda()
                        fake_A = self.G_BA(real_B)
                        fake_B = self.G_AB(real_A)
                        img_sample = torch.cat((
                            real_A
                            , fake_B
                            , real_B
                            , fake_A
                        ), 0)
                        save_image(
                            img_sample
                            , f'./progress_images/out_{epoch}_{idx}.png'
                            , nrow=self.test_dl.batch_size
                            , normalize=True
                        )

                    # print status
                    print(f'\nEpoch: {epoch}/{epochs} -- Batch: {idx}/{len(self.dl.dataset)}'
                          f'\nDiscriminator loss: {loss_D:.4f} Generator loss: {loss_G:.4f}'
                          f'\nTime taken: {time.time() - running_time:.4f}s')

                    running_time = time.time()

            # save models each epoch
            self.save_models(epoch)

    def evaluate(self, size):
        """
        Evaluates model
        :param size: size of eval image
        """
        # set eval flags
        self.G_AB.eval()
        self.G_BA.eval()
        self.D_A.eval()
        self.D_B.eval()

        with torch.no_grad():
            for idx, data in enumerate(self.test_dl):
                real_A = data['test_A'].cuda()
                real_B = data['test_B'].cuda()
                fake_A = self.G_BA(real_B)
                fake_B = self.G_AB(real_A)
                img_sample = torch.cat((
                    real_A
                    , fake_B
                    , real_B
                    , fake_A
                ), 0)
                save_image(
                    img_sample
                    , f'./eval/eval_{idx}.png'
                    , nrow=size
                    , normalize=True
                )







