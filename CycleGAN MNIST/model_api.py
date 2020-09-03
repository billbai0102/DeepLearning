import torch
from torch import optim
from torch.nn import init
from torchvision.utils import save_image

import os
import time
import numpy as np

from hyperparameters import *
from Generator import Generator
from Discriminator import Discriminator


class Model:
    def __init__(self, dl, classes, channels, img_sz, latent_dim):
        self.dl = dl
        self.classes = classes
        self.channels = channels
        self.img_sz = img_sz
        self.latent_dim = latent_dim

        self.gen = Generator(self.classes, self.channels, self.img_sz, self.latent_dim)
        self.dis = Discriminator(self.classes, self.channels, self.img_sz, self.latent_dim)
        self.gen.cuda()
        self.dis.cuda()

        self.gen_opt = None
        self.dis_opt = None

        self.history = {
            'gen_loss': [],
            'dis_loss': [],
        }

        self.eval_count = 0

        if not os.path.exists('./progress_images'):
            os.mkdir('./progress_images')
        if not os.path.exists('./models'):
            os.mkdir('./models')
        if not os.path.exists('./eval'):
            os.mkdir('./eval')

    @property
    def generator(self):
        return self.gen

    @property
    def discriminator(self):
        return self.dis

    def init_optimizer(self, lr, alpha=.888, beta=.999):
        """
        Initializes gen and dis optimizers
        :param lr: learning rate
        :param alpha: alpha
        :param beta: beta
        """
        # init generator optimizer
        self.gen_opt = optim.Adam(
            filter(
                lambda param: param.requires_grad
                , self.gen.parameters()
            )
            , lr=lr
            , betas=(alpha, beta)
        )
        # init discriminator optimizer
        self.dis_opt = optim.Adam(
            filter(
                lambda param: param.requires_grad
                , self.dis.parameters()
            )
            , lr=lr
            , betas=(alpha, beta)
        )

    def train(self, epochs):
        # set train flag to apply dropout and norm
        self.gen.train()
        self.dis.train()

        progress_noise = torch.randn(self.dl.batch_size, self.latent_dim).cuda()
        progress_label = torch.LongTensor(np.array([num for _ in range(self.dl.batch_size // 8) for num in range(8)])).cuda()

        # train loop
        for epoch in range(epochs):
            running_time = time.time()
            for idx, (data, target) in enumerate(self.dl):
                # move data and target to cuda
                data, target = data.cuda(), target.cuda()

                # initialize labels
                batch_sz = data.size(0)
                real_labels = torch.full((batch_sz, 1), REAL_LABEL).cuda()
                fake_labels = torch.full((batch_sz, 1), FAKE_LABEL).cuda()

                '''train generator'''
                self.gen.zero_grad()
                # create noise and labels
                noise = torch.randn(batch_sz, self.latent_dim).cuda()
                gen_labels = torch.randint(0, self.classes, (batch_sz, )).cuda()

                # feed noise and label into gen
                gen_out_fake = self.gen(noise=noise, labels=gen_labels)
                # feed gen out and label into dis
                dis_out_gen = self.dis(img=gen_out_fake, labels=gen_labels)

                # calculate gen loss
                gen_loss = self.dis.dis_loss(dis_out_gen, real_labels)
                gen_loss.backward()
                self.gen_opt.step()

                '''train discriminator'''
                # train on **real** data
                self.dis.zero_grad()
                dis_out_real = self.dis(data, target)
                dis_loss_real = self.dis.dis_loss(dis_out_real, real_labels)
                
                # train on **fake** data
                dis_out_fake = self.dis(gen_out_fake.detach(), gen_labels)
                dis_loss_fake = self.dis.dis_loss(dis_out_fake, fake_labels)

                # calculate average loss for real and fake data
                dis_loss = (dis_loss_real + dis_loss_fake) / 2
                dis_loss.backward()
                self.dis_opt.step()

                # status updates every half epoch
                if idx > 0 and batch_sz % (batch_sz / 2) == 0:
                    gen_loss = gen_loss.mean().item()
                    dis_loss = dis_loss.mean().item()

                    # store loss
                    self.history['gen_loss'].append(gen_loss)
                    self.history['dis_loss'].append(dis_loss)

                    # save models
                    self.save_model(epoch)

                    # save progress images
                    if epoch == 1:
                        save_image(data, './progress_images/real_sample.png', normalize=True)
                    with torch.no_grad():
                        progress_sample = self.gen(progress_noise, progress_label)
                        save_image(progress_sample, f'./progress_images/gen_sample_{epoch}.png', normalize=True)

                    # print status
                    print(f'\nEpoch: {epoch + 1}/{epoch} -- Batch: {idx}/{batch_sz}'
                          f'\nDiscriminator loss: {dis_loss:.4f}'
                          f'\nGenerator loss: {gen_loss:.4f}'
                          f'\nTime taken: {time.time() - running_time}')
                    running_time = time.time()

    def evaluate(self):
        # eval flag
        self.gen.eval()
        self.dis.eval()

        # generate labels
        batch_sz = self.dl.batch_size
        progress_noise = torch.randn(batch_sz, self.latent_dim, 1, 1).cuda()
        progress_labels = torch.LongTensor(np.array([num for _ in range(batch_sz // 8) for num in range(8)])).cuda()

        # evaluate
        with torch.no_grad():
            progress_sample = self.gen(progress_noise, progress_labels)
            progress_vector = progress_noise.detach().cpu().numpy().reshape(batch_sz, self.latent_dim)

            np.savetxt('latent_space_vec.txt')
            save_image(progress_sample, f'./eval/eval_{self.eval_count}.png', nrow=16, normalize=True)
            self.eval_count += 1

    def save_model(self, epoch):
        torch.save(self.dis.state_dict(), f'./models/dis_{epoch}')
        torch.save(self.gen.state_dict(), f'./models/gen_{epoch}')

    @classmethod
    def init_weights(cls, model):
        """
        Initializes weights for every conv and norm layer
        :param model: model to initialize weights
        """
        cls_name = model.__class__.__name__
        # initialize conv weights
        if cls_name.find('Conv') != -1:
            init.normal_(model.weight.data, 1., 2e-2)
        # initialize batchnorm weights
        if cls_name.find('BatchNorm') != 1:
            init.normal_(model.weight.data, 1., 2e-2)
            init.constant_(model.bias.data, 0.)

