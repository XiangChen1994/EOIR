import os
import glob
import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class oasis_pkl_loader(Dataset): # oasis_pkl_loader

    def __init__(self,
            root_dir = './../../../data/oasisreg',
            split = 'train', # train, val or test
            is_gen = 0,
        ):
        self.root_dir = root_dir
        self.split = split
        self.is_gen = is_gen

        train_dir = os.path.join(self.root_dir, 'train')
        val_dir = os.path.join(self.root_dir, 'val')
        self.total_list = []

        if self.split == 'train':
            self.total_list = glob.glob(os.path.join(train_dir,'*.pkl'))
        elif self.split == 'val' or self.split == 'test':
            self.total_list = glob.glob(os.path.join(val_dir,'*.pkl'))
        else:
            raise ValueError('Invalid split name')

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):

        fp = self.total_list[idx]

        if self.split == 'train':
            tar_list = self.total_list.copy()
            tar_list.remove(fp)
            random.shuffle(tar_list)
            tar_file = tar_list[0]
            x, x_seg = pkload(fp)
            y, y_seg = pkload(tar_file)
            x_idx = int(os.path.split(fp)[-1].split('.')[0].split('_')[1])
            y_idx = int(os.path.split(tar_file)[-1].split('.')[0].split('_')[1])
        elif self.split == 'val' or self.split == 'test':
            x, y, x_seg, y_seg = pkload(fp)
            sv_file_name = os.path.split(fp)[-1]
            x_idx, y_idx = sv_file_name.split('.')[0][2:].split('_')
            x_idx, y_idx = int(x_idx), int(y_idx)

        x, x_seg = x[None, ...], x_seg[None, ...]
        y, y_seg = y[None, ...], y_seg[None, ...]

        x, x_seg = np.ascontiguousarray(x), np.ascontiguousarray(x_seg)
        y, y_seg = np.ascontiguousarray(y), np.ascontiguousarray(y_seg)

        x, x_seg = torch.from_numpy(x), torch.from_numpy(x_seg)
        y, y_seg = torch.from_numpy(y), torch.from_numpy(y_seg)

        return x, x_seg, y, y_seg, idx, x_idx, y_idx
