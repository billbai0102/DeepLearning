import torch
from torch import nn
from torch import optim
from torchvision.utils import save_image

import os
import time
import numpy as np
from tqdm import tqdm

import ml_utils as u
from hyperparameters import *
from data import DataStream
from models import Generator
from models import Discriminator


class GAN:
    def __init__(self, dl, l1_c, l2_c):
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dl = dl
        self.l1_c = l1_c
        self.l2_c = l2_c

        self.latent_dim = LATENT_DIM
        self.embed_in_dim = EMBED_IN_DIM
        self.embed_out_dim = EMBED_OUT_DIM

        self.generator = Generator(self.latent_dim, self.embed_in_dim, self.embed_out_dim).to(self.device)
        self.discriminator = Discriminator(self.embed_out_dim, self.embed_in_dim).to(self.device)
        self.generator.apply(GAN._init_weights)
        self.discriminator.apply(GAN._init_weights)

        self.loss_adv = nn.BCELoss()
        self.loss_l1 = nn.L1Loss()
        self.loss_l2 = nn.MSELoss()

        self.optim_G = None
        self.optim_D = None
        self._init_optim(LR, BETAS)

        u.create_dir('./progress_imgs')
        u.create_dir('./eval')
        u.create_dir('./pt')

        self.cycles = 0
        self.hist = {
            'd_loss': [],
            'g_loss': []
        }

        u.log_data_to_txt('train_log', f'\nUsing device {self.device}')

    def _init_optim(self, lr, betas):
        self.optim_G = optim.Adam(
            u.filter_gradients(self.generator)
            , lr=lr
            , betas=betas
        )

        self.optim_D = optim.Adam(
            u.filter_gradients(self.discriminator)
            , lr=lr
            , betas=betas
        )

    def save_models(self):
        u.save_state_dict(self.generator, 'generator')
        u.save_state_dict(self.discriminator, 'discriminator')
        u.save_state_dict(self.optim_G, 'optim_g')
        u.save_state_dict(self.optim_D, 'optim_d')

    def train(self, epochs):
        self.generator.train()
        self.discriminator.train()
        for epoch in range(epochs):
            running_time = time.time()
            for idx, data in enumerate(self.dl):
                self.cycles += 1
                image = data['image'].to(self.device)
                embed = data['embed'].to(self.device)

                real_labels = torch.ones((image.shape[0])).to(self.device)
                fake_labels = torch.zeros((image.shape[0])).to(self.device)

                '''Train discriminator'''
                # on real data
                self.optim_D.zero_grad() # zero out gradients
                d_real, _ = self.discriminator(image, embed)
                d_loss_real = self.loss_adv(d_real, real_labels)

                # on fake data
                z = torch.randn((image.shape[0], 100, 1, 1)).to(self.device)
                g_fake = self.generator(z, embed)
                d_fake = self.discriminator(g_fake)
                d_loss_fake = self.loss_adv(g_fake, fake_labels)

                # calculate loss
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss_real.backward()
                self.optim_D.step()

                '''Train generator'''
                self.optim_G.zero_grad() # zero out gradients
                z = torch.randn((image.shape[0], 100, 1, 1)).to(self.device)
                g_fake = self.generator(z, embed)
                d_fake, act_fake = self.discriminator(g_fake, embed)
                _, act_real = self.discriminator(image, embed)

                # calculate loss
                g_loss_l1 = self.loss_l1(torch.mean(act_fake, 0), torch.mean(act_real, 0).detach())
                g_loss_l2 = self.loss_l2(g_fake, image)
                g_loss_adv = self.loss_adv(d_fake, real_labels)
                g_loss = g_loss_adv + (self.l1_c * g_loss_l1) + (self.l2_c * g_loss_l2)
                g_loss.backward()
                self.optim_G.step()

                '''Status updates'''
                if self.cycles % 100 == 0:
                    d_loss = d_loss.mean().item()
                    g_loss = g_loss.mean().item()

                    self.hist['d_loss'].append(d_loss)
                    self.hist['g_loss'].append(g_loss)

                    with torch.no_grad():
                        img_sample = torch.cat((image[:32]), 0)
                        save_image(img_sample
                                   , f'./progress_imgs/img_{epoch}_{idx}.png'
                                   , nrow=8
                                   , normalize=True)

                    u.log_data_to_txt('train_log',
                                      f'\nEpoch: {epoch}/{epochs} -- Batch: {idx}/{len(self.dl.dataset)}'
                                      f'\nDiscriminator Loss: {d_loss: .4f} Generator Loss: {g_loss:.4f}'
                                      f'\nTime taken: {time.time() - running_time:.4f}s')
                    running_time = time.time()

            self.save_models()

    @classmethod
    def _init_weights(cls, layer):
        name = layer.__class__.__name__
        if name.find('Conv') != -1:
            torch.nn.init.normal_(layer.weight.data, .0, 2e-2)
        if name.find('BatchNorm') != -1:
            torch.nn.init.normal_(layer.weight.data, 1.0, 2e-2)
            torch.nn.init.constant_(layer.bias.data, .0)
