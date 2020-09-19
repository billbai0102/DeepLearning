import torch
from torch.utils.data import Dataset, DataLoader

import io
import h5py
import numpy as np
from PIL import Image

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

    def __len__(self):
        f = h5py.File(self.file, 'r')
        self.dataset_keys = [str(key) for key in f[self.split].keys()]
        length = len(f[self.split])
        f.close()

        return length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.file, mode='r')
            self.dataset_keys = [str(key) for key in self.dataset[self.split].keys()]

        idx_name = self.dataset_keys[idx]
        data = self.dataset[self.split][idx_name]

        image = bytes(np.array(data['img']))
        image = Image.open(io.BytesIO(image)).resize((64, 64))
        image = self.preprocess(image)

        embed = np.array(data['embeddings'], dtype=float)
        txt = np.array(data['txt']).astype(str)

        item = {
            'image': torch.FloatTensor(image),
            'embed': torch.FloatTensor(embed),
            'txt': str(txt)
        }

        item['image'].sub_(255./2.).div_(255./2.)

        return item

    def preprocess(self, image):
        image = np.array(image, dtype=float)
        if len(image.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = image
            rgb[:, :, 1] = image
            rgb[:, :, 1] = image
            image = rgb

        return image.transpose(2, 0, 1)

