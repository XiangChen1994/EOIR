import os, glob
import torch, sys, json
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
import numpy as np
import nibabel as nib

class L2RLUMIRJSONDataset(Dataset):
    def __init__(self, base_dir, json_path, stage='train'):
        with open(json_path) as f:
            d = json.load(f)
        if stage.lower() == 'train':
            self.imgs = d['training']
        elif stage.lower() == 'validation':
            self.imgs = d['validation']
        elif stage.lower() == 'test':
            self.imgs = d['test']
        else:
            raise 'Not implemented!'
        self.base_dir = base_dir
        self.stage = stage

    def __getitem__(self, index):
        if self.stage == 'train':
            mov_dict = self.imgs[index]
            fix_dicts = self.imgs.copy()
            fix_dicts.remove(mov_dict)
            random.shuffle(fix_dicts)
            fix_dict = fix_dicts[0]
            x = nib.load(self.base_dir + mov_dict['image'])
            y = nib.load(self.base_dir + fix_dict['image'])
            x = x.get_fdata() / 255.
            y = y.get_fdata() / 255.
            x, y = x[None, ...], y[None, ...]
            x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)  # [channels,Height,Width,Depth]           
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            return x.float(), y.float()
        elif self.stage == 'validation':
            img_dict = self.imgs[index]
            img_path = img_dict['image']
            seg_path = img_dict['seg']
            x = nib.load(self.base_dir + img_path)
            x_seg = nib.load(self.base_dir + seg_path)
            x = x.get_fdata() / 255.
            x_seg = x_seg.get_fdata()
            x = x[None, ...]
            x_seg = x_seg[None, ...]
            x = np.ascontiguousarray(x)  # [channels,Height,Width,Depth]
            x_seg = np.ascontiguousarray(x_seg)
            x = torch.from_numpy(x)
            x_seg = torch.from_numpy(x_seg)
            return x.float(), x_seg.float()
        elif self.stage == 'test':
            img_dict = self.imgs[index]
            mov_path = img_dict['moving']
            fix_path = img_dict['fixed']
            x = nib.load(self.base_dir + mov_path)
            y = nib.load(self.base_dir + fix_path)
            x = x.get_fdata() / 255.
            y = y.get_fdata() / 255.
            x, y = x[None, ...], y[None, ...]
            x = np.ascontiguousarray(x)  # [channels,Height,Width,Depth]
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            return x.float(), y.float()

    def __len__(self):
        return len(self.imgs)