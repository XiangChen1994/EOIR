import os
import torch
import argparse
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from loaders import lumir_loader
from utils.functions import registerSTModel
from torch.autograd import Variable
from utils import getters, setters

def save_nii(img, file_name, pix_dim=[1., 1., 1.]):
    x_nib = nib.Nifti1Image(img, np.eye(4))
    x_nib.header.get_xyzt_units()
    x_nib.header['pixdim'][1:4] = pix_dim
    x_nib.to_filename('{}.nii.gz'.format(file_name))

def run(opt):
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)   
    # test save dir
    test_dir = './lumir_test/'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    # test dataset
    test_set = lumir_loader.L2RLUMIRJSONDataset(base_dir='./LUMIR_L2R24_TrainVal', 
                                                json_path='./LUMIR_L2R24_TrainVal/LUMIR_dataset.json',
                                                stage='test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    test_files = test_set.imgs
    # load model
    model, info = getters.getTestModelWithCheckpoints(opt)
    print('model_information:', info)
    # validation
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            moving_id = test_files[i]['moving'].split('_')[-2]
            fixed_id = test_files[i]['fixed'].split('_')[-2]
            model.eval()
            data = [Variable(t.cuda()) for t in data[:2]]
            fixed_img = data[1].cuda()
            moving_img = data[0].cuda()
            # rigistration : moving image ---> fixed image
            flow = model(moving_img, fixed_img, registration=True)
            flow = np.transpose(flow.detach().cpu().numpy()[0, :, :, :, :],(1,2,3,0))
            save_nii(flow, test_dir + 'disp_{}_{}'.format(fixed_id, moving_id))
            print('disp_{}_{}.nii.gz saved to {}'.format(fixed_id, moving_id, test_dir))

if __name__ == '__main__':
    opt = {
        'img_size': (160, 224, 192),    # input image size
        'logs_path': './logs',          # path to save log
        'log': './logs',
        'save_freq': 5,              # save model every save_freq epochs
        'n_checkpoints': 10,          # number of checkpoints to keep
        'power': 0.9,                # decay power
    }
    parser = argparse.ArgumentParser(description = "lumir")
    parser.add_argument("-m", "--model", type = str, default = 'EOIR')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'lumirreg')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("--num_workers", type = int, default = 4) # best, last or epoch
    parser.add_argument("--img_size", type = str, default = '(160, 224, 192)')
    parser.add_argument("--load_ckpt", type = str, default = "best") # best, last or epoch
    parser.add_argument("-dp", "--datasets_path", type = str, default = "./LUMIR_L2R24_TrainVal")
    parser.add_argument("--json_path", type = str, default = "./LUMIR_L2R24_TrainVal/LUMIR_dataset.json")
    parser.add_argument("--start_channel", type=str, default='16')

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    opt['nkwargs']['img_size'] = opt['img_size']
    opt['img_size'] = eval(opt['img_size'])

    run(opt)
    '''
    python test_registration_LUMIR.py --model EOIR --batch_size 1 --dataset lumirreg \
                                  --gpu_id 0 --num_workers 4 --load_ckpt best \
                                  --datasets_path ./LUMIR_L2R24_TrainVal \
                                  --json_path ./LUMIR_L2R24_TrainVal/LUMIR_dataset.json\
                                  --start_channel 32
    '''