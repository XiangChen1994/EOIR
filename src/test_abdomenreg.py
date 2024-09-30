import os
import torch
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from torch.autograd import Variable
import torch.nn.functional as F

from utils import getters, setters
from utils.mappers import label2text_dict_abdomenct as label2text_dict
from utils.functions import AverageMeter, registerSTModel, dice_eval, dice_binary, jacobian_determinant, compute_HD95
from monai.inferers import SlidingWindowInferer
from scipy.ndimage import zoom
import nibabel as nib

def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    model, _ = getters.getTestModelWithCheckpoints(opt)
    test_loader = getters.getDataLoader(opt, split=opt['field_split'])
    reg_model_ne = registerSTModel(opt['img_size'], 'nearest').cuda()
    reg_model_ti = registerSTModel(opt['img_size'], 'bilinear').cuda()

    organ_eval_dsc = [AverageMeter() for i in range(1,14)]
    eval_dsc = AverageMeter()
    init_dsc = AverageMeter()
    eval_det = AverageMeter()
    eval_std_det = AverageMeter()
    eval_hd95 = AverageMeter()
    init_hd95 = AverageMeter()
    df_data = []
    keys = ['idx1', 'idx2'] + [label2text_dict[i] for i in range(1,14)] + ['val_dice', 'init_dice', 'jac_det', 'std_dev', 'hd95', 'init_hd95']
    pid = 0
    with torch.no_grad():
        for num, data in enumerate(test_loader):
            model.eval()
            idx1, idx2 = data[4][0].item(), data[5][0].item()
            loop_df_data = [idx1, idx2]
            data = [Variable(t.cuda())  for t in data[:4]]
            x, x_seg = data[0], data[1]
            y, y_seg = data[2], data[3]

            #f_xy = model(x,y,x,y,registration=True)
            f_xy = model(x,y,registration=True)
            def_y = reg_model_ti(x, f_xy)
            def_out = reg_model_ne(x_seg.float(), f_xy)
            for idx in range(1,14):
                dsc_idx = dice_binary(def_out.long().squeeze().cpu().numpy(), y_seg.long().squeeze().cpu().numpy(), idx)
                loop_df_data.append(dsc_idx)
                organ_eval_dsc[idx-1].update(dsc_idx, x.size(0))
            dsc1 = dice_eval(def_out.long(), y_seg.long(), 14)
            eval_dsc.update(dsc1.item(), x.size(0))
            dsc2 = dice_eval(x_seg.long(), y_seg.long(), 14)
            init_dsc.update(dsc2.item(), x.size(0))

            jac_det = jacobian_determinant(f_xy.detach().cpu().numpy())
            jac_det_val = np.sum(jac_det <= 0) / np.prod(x_seg.shape)
            eval_det.update(jac_det_val, x.size(0))

            log_jac_det = np.log(np.abs((jac_det+3).clip(1e-8, 1e8)))
            std_dev_jac = np.std(log_jac_det)
            eval_std_det.update(std_dev_jac, x.size(0))

            moving = x_seg.long().squeeze().cpu().numpy()
            fixed = y_seg.long().squeeze().cpu().numpy()
            moving_warped = def_out.long().squeeze().cpu().numpy()
            hd95_1 = compute_HD95(moving, fixed, moving_warped,14,np.ones(3)*4)
            eval_hd95.update(hd95_1, x.size(0))
            hd95_2 = compute_HD95(moving, fixed, moving,14,np.ones(3)*4)
            init_hd95.update(hd95_2, x.size(0))

            print('idx1 {:d}, idx1 {:d}, val dice {:.4f}, init dice {:.4f}, jac det {:.4f}, std dev {:.4f}, hd95 {:.4f}, init hd95 {:.4f}'.format(idx1, idx2, dsc1.item(), dsc2.item(), jac_det_val, std_dev_jac, hd95_1, hd95_2))
            loop_df_data += [dsc1.item(), dsc2.item(), jac_det_val, std_dev_jac, hd95_1, hd95_2]
            df_data.append(loop_df_data)
            
            pid = pid + 1
            
            if num==35:
                print(num,'save')
                sv_x = x[0,0,:,:,:].cpu().numpy()
                sv_y = y[0,0,:,:,:].cpu().numpy()
                sv_xseg = x_seg[0,0,:,:,:].cpu().numpy()
                sv_yseg = y_seg[0,0,:,:,:].cpu().numpy()
                warp_x = def_y[0,0,:,:,:].cpu().numpy()
                warpd_xseg = def_out[0,0,:,:,:].long().float().cpu().numpy()
                flow_x2y = f_xy #F.interpolate(f_xy, scale_factor=2., mode='trilinear', align_corners=True)
                flow_x2y = flow_x2y.permute(2,3,4,0,1).cpu().numpy()

                fp = os.path.join(opt['log'],'%s_%s_reg' % (str(idx1).zfill(4), str(idx2).zfill(4)))
                os.makedirs(fp, exist_ok = True)

                nib.save(nib.Nifti1Image(sv_x, None, None), os.path.join(fp,'img_moving.nii.gz'))
                nib.save(nib.Nifti1Image(sv_y, None, None), os.path.join(fp,'img_fixed.nii.gz'))
                nib.save(nib.Nifti1Image(sv_xseg, None, None), os.path.join(fp,'seg_moving.nii.gz'))
                nib.save(nib.Nifti1Image(sv_yseg, None, None), os.path.join(fp,'seg_fixed.nii.gz'))
                nib.save(nib.Nifti1Image(warp_x, None, None), os.path.join(fp,'img_moving_warp.nii.gz'))
                nib.save(nib.Nifti1Image(warpd_xseg, None, None), os.path.join(fp,'seg_moving_warped.nii.gz'))
                nib.save(nib.Nifti1Image(flow_x2y, None, None), os.path.join(fp,'flow_x2y.nii.gz'))
            #

    avg_organ_eval_dsc = [organ_eval_dsc[i].avg for i in range(13)]
    avg_df_data = [0,0] + avg_organ_eval_dsc + [eval_dsc.avg, init_dsc.avg, eval_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg]
    df_data.append(avg_df_data)

    print('Avg val dice {:.4f}, Avg init dice {:.4f}, Avg jac det {:.4f}, Avg std dev {:.4f}, Avg hd95 {:.4f}, Avg init hd95 {:.4f}'.format(eval_dsc.avg, init_dsc.avg, eval_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg))

    df = pd.DataFrame(df_data, columns=keys)
    fp = os.path.join('logs', opt['dataset'], 'results_%s.csv' % opt['model'])
    df.to_csv(fp, index=False)

if __name__ == '__main__':

    opt = {
        'logs_path': './logs',
        'save_freq': 5,
        'n_checkpoints': 2,
        'num_workers': 4,
        'in_shape': (192//2,160//2,256//2),
    }
    
    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("-m", "--model", type = str, default = 'VxmDense')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'abdomenreg')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "./../../../data/")
    parser.add_argument("--load_ckpt", type = str, default = "best") # best, last or epoch
    parser.add_argument("--field_split", type = str, default = 'test')
    parser.add_argument("--img_size", type = str, default = '(192//2,160//2,256//2)')
    parser.add_argument("--fea_type", type = str, default = 'unet')

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    #opt['nkwargs']['img_size'] = opt['img_size']
    opt['img_size'] = eval(opt['img_size'])
    

    run(opt)
    '''
    python test_abdomenreg.py -m EOIR -d abdomenreg -bs 1 start_channel=32 --gpu_id 5
    '''
