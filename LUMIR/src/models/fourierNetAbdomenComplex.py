import math
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.backbones.voxelmorph.torch import layers
from torch.distributions.normal import Normal


class SYMNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(SYMNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.r_dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)

        self.rr_dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.r_up1 = self.decoder(self.start_channel*8,self.start_channel*8,kernel_size=2,stride=2)
        self.r_up2 = self.decoder(self.start_channel*4,self.start_channel*4,kernel_size=2,stride=2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))#,
                # nn.Softsign())
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        r_d0 = torch.cat((self.r_up1(e4), e3), 1)

        r_d0 = self.r_dc1(r_d0)
        r_d0 = self.r_dc2(r_d0)

        r_d1 = torch.cat((self.r_up2(r_d0), e2), 1)

        r_d1 = self.r_dc3(r_d1)
        r_d1 = self.r_dc4(r_d1)

        f_r = self.rr_dc9(r_d1)

        return f_r


class fourierNetAbdomenComplex(nn.Module):

    def __init__(self, 
        in_channel=2,
        n_classes=3, 
        start_channel='32',
        img_size='(96,80,128)', # (128,128,16) for ACDC
    ):
        super(fourierNetAbdomenComplex, self).__init__()

        self.start_channel = int(start_channel)
        self.img_size = eval(img_size)

        print("start_channel: %d, img_size: %s" % (self.start_channel, str(self.img_size)))

        self.out_size = (self.img_size[0]//4,self.img_size[1]//4,self.img_size[2]//4)
        pd_size = [(is_-os_) // 2 for is_,os_ in zip(self.img_size,self.out_size)]
        self.p3d = [pd_size[2],pd_size[2],pd_size[1],pd_size[1],pd_size[0],pd_size[0]]
        self.base_net = SYMNet(in_channel, n_classes, self.start_channel)

        self.transformer = layers.SpatialTransformer(self.img_size)

    def forward(self, x, y, x_seg=None, y_seg=None, registration=False):

        out= self.base_net(x, y)
        fft_dims = (2,3,4)
        out_ifft = torch.fft.fftshift(torch.fft.fftn(out,dim=fft_dims),dim=fft_dims)
        out_ifft = F.pad(out_ifft, self.p3d , "constant", 0)
        f_xy = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft,dim=fft_dims),dim=fft_dims))
        source, pos_flow = x, f_xy
        pos_flow[:,0,...] = pos_flow[:,0,...] * (self.img_size[0] - 1)
        pos_flow[:,1,...] = pos_flow[:,1,...] * (self.img_size[1] - 1)
        pos_flow[:,2,...] = pos_flow[:,2,...] * (self.img_size[2] - 1)
        preint_flow = pos_flow
        y_source = self.transformer(source, pos_flow)

        if not registration:
            return y_source, preint_flow
        else:
            return y_source, pos_flow
'''
python train_abdomenreg_scp.py -d abdomenreg -m fourierNetAbdomenComplex -bs 1 --epochs 101
'''