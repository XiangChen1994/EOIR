import torch 
import torch.nn as nn
from torch.distributions.normal import Normal

from models.backbones.layers import encoder, decoder
from models.backbones.voxelmorph.torch.layers import SpatialTransformer, VecInt


class dispWarp(nn.Module):

    def __init__(self, in_cs, out_cs=3, lk_size=7, act=nn.Identity, is_int=0, img_size=None):

        super(dispWarp, self).__init__()

        self.img_size = img_size
        self.is_int = is_int

        self.disp_field = nn.Sequential(
            encoder(2*in_cs, in_cs, lk_size, 1, lk_size//2),
            nn.Conv3d(in_cs, out_cs, 3, 1, 1),
            act(),
        )

        self.up_tri = torch.nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)
        if self.img_size is not None and self.is_int:
            self.integrate = VecInt(self.img_size, 7)

    def forward(self, e, d, st_e, st_d, up_field=None):

        e_x, e_y = torch.chunk(e, 2, dim=0)
        d_x, d_y = torch.chunk(d, 2, dim=0)

        field = self.disp_field(torch.cat((d_y+d_x, d_y-d_x), dim=1)) / 2
        preint_field = field
        #print(field.shape)
        #input()
        if self.is_int and self.img_size is not None:
            field = self.integrate(field)

        warped_d_x = st_d(d_x, field)
        d = torch.cat((warped_d_x, d_y), dim=0)

        if up_field is not None:
            field = field + up_field

        up_field = self.up_tri(field) * 2

        warped_e_x = st_e(e_x, up_field)
        e = torch.cat((warped_e_x, e_y), dim=0)

        return e, d, preint_field, field, up_field

class someWarpComplex(nn.Module):

    def __init__(self,
            start_channel = '12',      # N_s in the paper
            is_int = '1',              # whether to use integration
            lk_size = '5',             # kernel size of LK encoder
            img_size = '(128,128,16)', # input image size
        ):

        super(someWarpComplex, self).__init__()

        self.start_channel = int(start_channel)
        self.is_int = int(is_int)
        self.lk_size = int(lk_size)
        self.img_size = eval(img_size)

        print("start_channel: %d, is_int: %d, lk_size: %d, img_size: %s" % (self.start_channel, self.is_int, self.lk_size, self.img_size))

        N_s = self.start_channel
        ks = self.lk_size
        ss = self.img_size

        self.eninput = encoder(1, N_s)
        self.ec1 = encoder(N_s, N_s)
        self.ec2 = encoder(N_s, N_s * 2, 3, 2, 1) # stride=2
        self.ec3 = encoder(N_s * 2, N_s * 2, ks, 1, ks//2) # LK encoder
        self.ec4 = encoder(N_s * 2, N_s * 4, 3, 2, 1) # stride=2
        self.ec5 = encoder(N_s * 4, N_s * 4, ks, 1, ks//2) # LK encoder
        self.ec6 = encoder(N_s * 4, N_s * 8, 3, 2, 1) # stride=2
        self.ec7 = encoder(N_s * 8, N_s * 8, ks, 1, ks//2) # LK encoder
        self.ec8 = encoder(N_s * 8, N_s * 8, 3, 2, 1) # stride=2
        self.ec9 = encoder(N_s * 8, N_s * 8, ks, 1, ks//2) # LK encoder

        self.dc1 = encoder(N_s * 16, N_s * 8, 3, 1, 1)
        self.dc2 = encoder(N_s * 8,  N_s * 4, ks, 1, ks//2)
        self.dc3 = encoder(N_s * 8,  N_s * 4, 3, 1, 1)
        self.dc4 = encoder(N_s * 4,  N_s * 2, ks, 1, ks//2)
        self.dc5 = encoder(N_s * 4,  N_s * 4, 3, 1, 1)
        self.dc6 = encoder(N_s * 4,  N_s * 2, ks, 1, ks//2)
        self.dc7 = encoder(N_s * 3,  N_s * 2, 3, 1, 1)
        self.dc8 = encoder(N_s * 2,  N_s * 2, ks, 1, ks//2)

        self.up1 = decoder(N_s * 8, N_s * 8, kernel_size=2,stride=2)
        self.up2 = decoder(N_s * 4, N_s * 4, kernel_size=2,stride=2)
        self.up3 = decoder(N_s * 2, N_s * 2, kernel_size=2,stride=2)
        self.up4 = decoder(N_s * 2, N_s * 2, kernel_size=2,stride=2)

        self.disp_warp_4 = dispWarp(N_s * 8, 3, ks, nn.Identity, self.is_int, [s // 16 for s in ss])
        self.disp_warp_3 = dispWarp(N_s * 4, 3, ks, nn.Identity, self.is_int, [s // 8 for s in ss])
        self.disp_warp_2 = dispWarp(N_s * 2, 3, ks, nn.Identity, self.is_int, [s // 4 for s in ss])
        self.disp_warp_1 = dispWarp(N_s * 2, 3, ks, nn.Identity, self.is_int, [s // 2 for s in ss])
        self.disp_warp_0 = nn.Sequential(
            encoder(N_s * 4, N_s * 2, ks, 1, ks//2),
            nn.Conv3d(N_s * 2, 3, 3, 1, 1),
            nn.Identity(),
        )

        self.transformer_5 = SpatialTransformer([s // 16 for s in ss])
        self.transformer_4 = SpatialTransformer([s // 8 for s in ss])
        self.transformer_3 = SpatialTransformer([s // 4 for s in ss])
        self.transformer_2 = SpatialTransformer([s // 2 for s in ss])
        self.transformer_1 = SpatialTransformer([s // 1 for s in ss])

        self.up_tri = torch.nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True)

    def forward(self, x, y, registration=False):

        x_in = torch.cat((x, y), dim=0)

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
        e3,e4,preint_field_4,field_4,up_field_4 = self.disp_warp_4(e3,e4,self.transformer_4,self.transformer_5)

        d0 = torch.cat((self.up1(e4), e3), dim=1)
        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        e2,d0,preint_field_3,field_3,up_field_3 = self.disp_warp_3(e2,d0,self.transformer_3,self.transformer_4,up_field_4)

        d1 = torch.cat((self.up2(d0), e2), dim=1)
        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        e1,d1,preint_field_2,field_2,up_field_2 = self.disp_warp_2(e1,d1,self.transformer_2,self.transformer_3,up_field_3)

        d2 = torch.cat((self.up3(d1), e1), dim=1)
        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        e0,d2,preint_field_1,field_1,up_field_1 = self.disp_warp_1(e0,d2,self.transformer_1,self.transformer_2,up_field_2)

        d3 = torch.cat((self.up4(d2), e0), dim=1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)
        d3_x, d3_y = torch.chunk(d3, 2, dim=0)
        field_0 = self.disp_warp_0(torch.cat((d3_y+d3_x, d3_y-d3_x), dim=1)) / 2
        preint_field_0 = field_0
        field_0 = field_0 + up_field_1

        int_flows = [preint_field_0,preint_field_1,preint_field_2,preint_field_3,preint_field_4]
        pos_flows = [field_0,field_1,field_2,field_3,field_4]

        if not registration:
            return int_flows, pos_flows
        else:
            return pos_flows[0]