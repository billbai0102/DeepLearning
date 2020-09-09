import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

import os
import random
from glob import glob
from PIL import Image


class CycleGANDataset(Dataset):
    def __init__(self, transform, mode='train'):
        super(Dataset, self).__init__()
        self.transform = transform
        self.train = (mode == 'train')

        self.path_A = sorted(glob(os.path.join('./data/vangogh2photo', '%sA' % mode) + '/*.*'))
        self.path_B = sorted(glob(os.path.join('./data/vangogh2photo', '%sB' % mode) + '/*.*'))

    def __getitem__(self, idx):
        item_A = self.transform(Image.open(self.path_A[idx % len(self.path_A)]))
        item_B = self.transform(Image.open(self.path_A[idx % len(self.path_A)]))

        if self.train:
            return {
                'train_A': item_A,
                'train_B': item_B
            }
        else:
            return {
                'test_A': item_A,
                'test_B': item_B
            }

    def __len__(self):
        return max(len(self.path_A), len(self.path_B))
