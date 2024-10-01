import os
import torch
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from torch.autograd import Variable
import torch.nn.functional as F

from utils import getters, setters
from utils.functions import AverageMeter, registerSTModel
from scipy.ndimage.interpolation import zoom

def convert_pytorch_grid2scipy(grid):
    _, H, W, D = grid.shape
    grid_x = (grid[0, ...] + 1) * (D -1)/2
    grid_y = (grid[1, ...] + 1) * (W -1)/2
    grid_z = (grid[2, ...] + 1) * (H -1)/2
    grid = np.stack([grid_z, grid_y, grid_x])
    identity_grid = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    grid = grid - identity_grid
    return grid

def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    test_loader = getters.getDataLoader(opt, split=opt['field_split'])
    reg_model_final = registerSTModel(opt['img_size'], 'nearest').cuda()
    model, _ = getters.getTestModelWithCheckpoints(opt)

    with torch.no_grad():
        for data_ori in test_loader:
            model.eval()
            data = [Variable(t.cuda())  for t in data_ori[:4]]
            x, x_seg = data[0].float(), data[1].long()
            y, y_seg = data[2].float(), data[3].long()
            sample_idx = data_ori[4]
            _, sv_file_name = os.path.split(test_loader.dataset.total_list[sample_idx])
            sub_idxs = sv_file_name.split('.')[0][2:]
            sub1_idx, sub2_idx = sub_idxs.split('_')

            # x -> y
            pos_flow = model(x,y,registration=True)

            flow = pos_flow
            print('Flow shape: %s, x shape: %s' % (str(flow.shape), str(x.shape)))
            _, pytorch_grid = reg_model_final(x, flow, is_grid_out=True)
            pytorch_grid = pytorch_grid.squeeze(0).permute(3,0,1,2).contiguous() # (3,h,w,d)
            scipy_disp = convert_pytorch_grid2scipy(pytorch_grid.data.cpu().numpy())

            downsample_scipy_disp = np.array([zoom(scipy_disp[i], 0.5, order=2) for i in range(3)])
            print('downsample_scipy_disp shape: %s' % (str(downsample_scipy_disp.shape)))
            fp = os.path.join(opt['log'], 'task_03')
            os.makedirs(fp, exist_ok=True)
            file_name = 'disp_'+sv_file_name.split('.')[0][2:] + '.npz'
            fp1 = os.path.join(fp, file_name)
            np.savez(fp1, np.array(downsample_scipy_disp).astype(np.float16))

if __name__ == '__main__':

    opt = {
        'logs_path': './logs',
        'save_freq': 5,
        'n_checkpoints': 2,
        'num_workers': 4,
        'img_size': (160//2,192//2,224//2),
    }

    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("-m", "--model", type = str, default = 'VxmDense')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'abdomenreg')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "./../../../data/")
    parser.add_argument("--load_ckpt", type = str, default = "best") # best, last or epoch
    parser.add_argument("--field_split", type = str, default = 'test')
    parser.add_argument("--img_size", type = str, default = '(160,192,224)')
    parser.add_argument("--fea_type", type = str, default = 'unet')

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    print(unknowns)
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    opt['nkwargs']['img_size'] = opt['img_size']
    opt['img_size'] = eval(opt['img_size'])

    run(opt)

'''
python test_abdomenreg.py -m encoderOnlyComplexS32 -d abdomenreg -bs 1 start_channel=32
'''
