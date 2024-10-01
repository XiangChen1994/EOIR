import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models.backbones.voxelmorph.torch.layers import SpatialTransformer

from utils.functions import AverageMeter, registerSTModel, adjust_learning_rate, get_downsampled_images, dice_val_VOI
from utils.loss import Grad3d, NccLoss
from utils import getters, setters

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 224, 192)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    # Getting model-related components
    train_loader = getters.getDataLoader(opt, split = 'train')
    val_loader = getters.getDataLoader(opt, split = 'validation')
    
    model, optimizer, init_epoch = getters.getTrainModelWithCheckpoints(opt)
    model_saver = getters.getModelSaver(opt)

    reg_model = registerSTModel(opt['img_size'], 'nearest').cuda()
    spatial_trans = SpatialTransformer((160, 224, 192)).cuda()

    criterion_reg = Grad3d(penalty = 'l2')
    criterion_ncc = NccLoss()
    best_dsc = 0
    best_epoch = 0

    for epoch in range(init_epoch, opt['epochs']):
        '''
        Training
        '''
        time_epoch = time.time()
        loss_all = AverageMeter()
        loss_reg_all = AverageMeter()
        loss_ncc_all = AverageMeter()
    
        for idx, data in enumerate(train_loader):
            model.train()
            adjust_learning_rate(optimizer, epoch, opt['epochs'], opt['lr'], opt['power'])
            data = [Variable(t.cuda()) for t in data[:2]]
            x = data[0].float()
            y = data[1].float()

            xs = get_downsampled_images(x, 4, mode = 'trilinear')
            ys = get_downsampled_images(y, 4, mode = 'trilinear')
            
            int_flows, pos_flows = model(x, y)

            reg_loss_0 = criterion_reg(int_flows[0])
            reg_loss_1 = criterion_reg(int_flows[1]) / 2
            reg_loss_2 = criterion_reg(int_flows[2]) / 4
            reg_loss_3 = criterion_reg(int_flows[3]) / 8
            reg_loss_4 = criterion_reg(int_flows[4]) / 16
            reg_loss = (reg_loss_0 + reg_loss_1 + reg_loss_2 + reg_loss_3 + reg_loss_4) * opt['reg_w']
            loss_reg_all.update(reg_loss.item(), y.numel())
    
            # NCC loss
            ncc_loss_0 = criterion_ncc(model.transformers[0](xs[0], pos_flows[0]), ys[0])
            ncc_loss_1 = criterion_ncc(model.transformers[1](xs[1], pos_flows[1]), ys[1]) / 2
            ncc_loss_2 = criterion_ncc(model.transformers[2](xs[2], pos_flows[2]), ys[2]) / 4
            ncc_loss_3 = criterion_ncc(model.transformers[3](xs[3], pos_flows[3]), ys[3]) / 8
            ncc_loss_4 = criterion_ncc(model.transformers[4](xs[4], pos_flows[4]), ys[4]) / 16
            ncc_loss = (ncc_loss_0 + ncc_loss_1 + ncc_loss_2 + ncc_loss_3 + ncc_loss_4) * opt['ncc_w']
            loss_ncc_all.update(ncc_loss.item(), y.numel())
            
            loss = reg_loss + ncc_loss

            output = spatial_trans(x, pos_flows[0])
            
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iter {} of {} loss {:.4f}, Reg {:.4f}, Ncc {:.4f}'.format(idx, len(train_loader), loss.item(), reg_loss.item(), ncc_loss.item()), end='\r', flush=True)
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch [{}/{}], Time {:.2f}, Loss {:.4f}, Reg {:.4f}, Ncc {:.4f}'.format(epoch, opt['epochs'], time.time()-time_epoch, loss_all.avg, loss_reg_all.avg, loss_ncc_all.avg))
        # tensorboard
        grid_img = mk_grid_img(8, 1)
        def_grid = spatial_trans(grid_img.float(), pos_flows[0].cuda())
        plt.switch_backend('agg')
        pred_fig = comput_fig(output)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x)
        tar_fig = comput_fig(y)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('moving', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('fixed', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('deformed', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
        
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        init_dsc = AverageMeter()
        with torch.no_grad():
            data = list(val_loader)
            for i in range(len(data)):
                fixed_img, fixed_label = data[i]
                fixed_img, fixed_label = fixed_img.cuda(), fixed_label.cuda()
                for j in range(len(data)):
                    if i != j:
                        moving_img, moving_label = data[j]
                        moving_img, moving_label = moving_img.cuda(), moving_label.cuda()
                        dsc = dice_val_VOI(fixed_label.long(), moving_label.long())
                        init_dsc.update(dsc.item(), fixed_img.size(0))
                        # rigistration:x-->y
                        pos_flow = model(moving_img, fixed_img, registration=True)
                        def_out = reg_model(moving_label.float(), pos_flow)
                        dsc = dice_val_VOI(def_out.long(), fixed_label.long())
                        eval_dsc.update(dsc.item(), fixed_img.size(0))
                        print('registration[img{}-->img{}], dsc {:.6f}'.format(j, i, dsc), end='\r', flush=True)

        if eval_dsc.avg > best_dsc:
            best_dsc = eval_dsc.avg
            best_epoch = epoch
        writer.add_scalar('Val/dice', eval_dsc.avg, epoch)
        print('Epoch [{}/{}], Time {:.6f}, init DSC {:.6f}, eval DSC {:.6f}, best DSC {:.6f} at epoch {}'.format(epoch, opt['epochs'], time.time()-time_epoch, init_dsc.avg, eval_dsc.avg, best_dsc, best_epoch))

        model_saver.saveModel(model, optimizer, epoch, eval_dsc.avg)
    writer.close()

if __name__ == '__main__':

    tensorboard_log_dir = './tensorboard_log/'
    writer = SummaryWriter(log_dir = tensorboard_log_dir)

    opt = {
        'img_size': (160, 224, 192),  # input image size
        'logs_path': './logs/',       # path to save logs
        'save_freq': 5,              # save model every save_freq epochs
        'n_checkpoints': 10,         # number of checkpoints to keep
        'power': 0.9,                # decay power
    }

    parser = argparse.ArgumentParser(description = "Brain")
    parser.add_argument("-m", "--model", type = str, default = 'EOIR')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'lumir')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "./LUMIR_L2R24_TrainVal")
    parser.add_argument("--json_path", type = str, default = "./LUMIR_L2R24_TrainVal/LUMIR_dataset.json")
    parser.add_argument("--epochs", type = int, default = 301)
    parser.add_argument("--reg_w", type = float, default = 5.)
    parser.add_argument("--ncc_w", type = float, default = 1.)
    parser.add_argument("--lr", type = float, default = 4e-4)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--img_size", type = str, default = '(160, 224, 192)')
    parser.add_argument("--start_channel", type = int, default = 32)
    parser.add_argument("--is_bidir", type = int, default = 0) 

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    opt['nkwargs']['img_size'] = opt['img_size']
    opt['img_size'] = eval(opt['img_size'])

    print('ncc_w: %.4f, reg_w: %.4f, img_size: %s' % (opt['ncc_w'], opt['reg_w'], opt['img_size']))

    run(opt)
    '''
    python train_registration_LUMIR.py --model EOIR --batch_size 1 --dataset lumirreg --gpu_id 0 \
                                   --epochs 301 --reg_w 5.0 --ncc_w 1.0 --lr 4e-4 \
                                   --start_channel 32 \
                                   --json_path ./LUMIR_L2R24_TrainVal/LUMIR_dataset.json \
                                   --datasets_path ./LUMIR_L2R24_TrainVal
    '''