import torch
from torch.utils.data import Dataset, DataLoader

import h5py
import numpy as np

class DataStream:
    def __init__(self, dl):
        self.dl = iter(dl)
        self.stream = torch.cuda.Stream()
        self.next_input = None
        self.next_target = None
        self.load_next_data()

    def load_next_data(self):
        try:
            self.next_input, next_target = next(self.dl)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        cur_input = self.next_input
        cur_target = self.next_target
        self.load_next_data()

        return cur_input, cur_target


class StackGANDataset(Dataset):
    def __init__(self, file, transform, split='train'):
        self.file = file
        self.transform = transform
        self.split = split

        self.dataset = None
        self.dataset_keys = None

        self.h5_int = lambda x: int(np.array(x))

    def __len__(self):
        f = h5py.File(self.file, 'r')
        self.dataset_keys = [str(key) for key in f[self.split].keys()]
        length = len(f[self.split])
        f.close()

        return length

    def __getitem__(self, idx):

