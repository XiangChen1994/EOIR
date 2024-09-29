import re
import os
import glob
import scipy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from collections import deque, OrderedDict
from models.backbones.transmorph.surface_distance import compute_robust_hausdorff, compute_surface_distances

def get_downsampled_images(img, n_downs=4, mode='trilinear', n_cs=1):

    blur = GaussianBlur3D(n_cs, sigma=1).to(img.device)
    out_imgs = [img]
    for _ in range(n_downs):
        img = blur(img)
        img = F.interpolate(img, scale_factor=0.5, mode=mode, align_corners=True)
        out_imgs.append(img)

    return out_imgs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, is_grid_out=False, mode=None, align_corners=True):
        #print('src:', src.shape)
        #print('flow:', flow.shape)
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if mode is None:
            out = F.grid_sample(src, new_locs, align_corners=align_corners, mode=self.mode)
        else:
            out = F.grid_sample(src, new_locs, align_corners=align_corners, mode=mode)

        if is_grid_out:
            return out, new_locs
        return out

class registerSTModel(nn.Module):

    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(registerSTModel, self).__init__()

        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, img, flow, is_grid_out=False, align_corners=True):

        out = self.spatial_trans(img,flow,is_grid_out=is_grid_out,align_corners=align_corners)

        return out

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)
    x = param_group['lr']
    return x

def dice_val_VOI(y_pred, y_true):
    VOI_lbls = [  0,   4,  11,  23,  30,  31,  32,  35,  36,  37,  38,  39,  40,  41,
                 44,  45,  47,  48,  49,  50,  51,  52,  55,  56,  57,  58,  59,  60,
                 61,  62,  71,  72,  73,  75,  76, 100, 101, 102, 103, 104, 105, 106,
                107, 108, 109, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                123, 124, 125, 128, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140,
                141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
                155, 156, 157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
                171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
                185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
                201, 202, 203, 204, 205, 206, 207]
    pred = y_pred[0,0,...]
    true = y_true[0,0,...]
    DSCs = torch.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = (pred == i).float()
        true_i = (true == i).float()
        intersection = pred_i * true_i
        intersection = torch.sum(intersection)
        union = torch.sum(pred_i) + torch.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] = dsc
        idx += 1
    return torch.mean(DSCs)

class modelSaver():

    def __init__(self, save_path, save_freq, n_checkpoints = 10):

        self.save_path = save_path
        self.save_freq = save_freq
        self.best_score = -1e6
        self.best_loss = 1e6
        self.n_checkpoints = n_checkpoints
        self.epoch_fifos = deque([])
        self.score_fifos = deque([])
        self.loss_fifos = deque([])

        self.initModelFifos()

    def initModelFifos(self):

        epoch_epochs = []
        score_epochs = []
        loss_epochs  = []

        file_list = glob.glob(os.path.join(self.save_path, '*epoch*.pth'))
        if file_list:
            for file_ in file_list:
                file_name = "net_epoch_(.*)_score_.*.pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    epoch_epochs.append(int(result[0]))

                file_name = "best_score_.*_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    score_epochs.append(int(result[0]))

                file_name = "best_loss_.*_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    loss_epochs.append(int(result[0]))

        score_epochs.sort()
        epoch_epochs.sort()
        loss_epochs.sort()

        if file_list:
            for file_ in file_list:
                for epoch_epoch in epoch_epochs:
                    file_name = "net_epoch_" + str(epoch_epoch) + "_score_.*.pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.epoch_fifos.append(result[0])

                for score_epoch in score_epochs:
                    file_name = "best_score_.*_net_epoch_" + str(score_epoch) +".pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.score_fifos.append(result[0])

                for loss_epoch in loss_epochs:
                    file_name = "best_loss_.*_net_epoch_" + str(loss_epoch) +".pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.loss_fifos.append(result[0])

        print("----->>>> BEFORE: epoch_fifos length: %d, score_fifos_length: %d, loss_fifos_length: %d" % (len(self.epoch_fifos), len(self.score_fifos), len(self.loss_fifos)))

        self.updateFIFOs()

        print("----->>>> AFTER: epoch_fifos length: %d, score_fifos_length: %d, loss_fifos_length: %d" % (len(self.epoch_fifos), len(self.score_fifos), len(self.loss_fifos)))

    def saveModel(self, model, optimizer, epoch, avg_score, loss=None):

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(self.save_path, 'net_latest.pth'))

        if epoch % self.save_freq == 0:

            file_name = ('net_epoch_%d_score_%.4f.pth' % (epoch, avg_score))
            self.epoch_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)

        if avg_score >= self.best_score:

            self.best_score = avg_score
            file_name = ('best_score_%.4f_net_epoch_%d.pth' % (avg_score, epoch))
            self.score_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)

        if loss is not None and loss <= self.best_loss:

            self.best_loss = loss
            file_name = ('best_loss_%.4f_net_epoch_%d.pth' % (loss, epoch))
            self.loss_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)

        self.updateFIFOs()

    def updateFIFOs(self):

        while(len(self.epoch_fifos) > self.n_checkpoints):
            file_name = self.epoch_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        while(len(self.score_fifos) > self.n_checkpoints):
            file_name = self.score_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        while(len(self.loss_fifos) > self.n_checkpoints):
            file_name = self.loss_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

def convert_state_dict(state_dict, is_multi = False):

    new_state_dict = OrderedDict()

    if is_multi:
        if next(iter(state_dict)).startswith("module."):
            return state_dict  # abort if dict is a DataParallel model_state

        for k, v in state_dict.items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
    else:

        if not next(iter(state_dict)).startswith("module."):
            return state_dict  # abort if dict is not a DataParallel model_state

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

    return new_state_dict

class GaussianBlur3D(nn.Module):
    def __init__(self, channels, sigma=1, kernel_size=0):
        super(GaussianBlur3D, self).__init__()
        self.channels = channels
        if kernel_size == 0:
            kernel_size = int(2.0 * sigma * 2 + 1)

        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size**2).view(kernel_size, kernel_size, kernel_size)
        y_grid = x_grid.transpose(0, 1)
        z_grid = x_grid.transpose(0, 2)
        xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.

        gaussian_kernel = (1. / (2. * np.pi * variance) ** 1.5) * \
                          torch.exp(
                              -torch.sum((xyz_grid - mean) ** 2., dim=-1) /
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding = kernel_size // 2

    def forward(self, x):
        blurred = F.conv3d(x, self.gaussian_kernel, padding=self.padding, groups=self.channels)
        return blurred
