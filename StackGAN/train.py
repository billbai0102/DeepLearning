import torch
from torch.utils.data import DataLoader

import numpy as np
import argparse
import hyperparameters as h

import ml_utils as u
from model_api import GAN
from data import StackGANDataset

args = None

def main():
    ds = StackGANDataset('./data/birds.hdf5', transform=None, split=h.MODE)
    dl = DataLoader(ds, batch_size=h.BATCH_SIZE)

    model = GAN(dl, h.L1_C, h.L2_C)

    model.train(h.EPOCHS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StackGAN')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--l1_coefficient', type=int, default=50, help='L1 multiplier')
    parser.add_argument('--l2_coefficient', type=int, default=100, help='L2 multiplier')
    parser.add_argument('--channels', type=int, default=3, help='Image channels (RGB = 3)')
    parser.add_argument('--seed', type=int, default=321, help='PyTorch seed')
    parser.add_argument('--train', type=u.str2bool, default=False, help='train')
    parser.add_argument('--eval', type=u.str2bool, default=False, help='eval')

    # set hyperparameters
    args = parser.parse_args()
    h.EPOCHS = args.epochs
    h.BATCH_SIZE = args.batch_size
    h.LR = args.learning_rate
    h.L1_C = args.l1_coefficient
    h.L2_C = args.l2_coefficient
    h.CHANNELS = args.channels
    seed = args.seed
    h.MODE = u.check_mode(args.train, args.eval)

    # for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    main()
