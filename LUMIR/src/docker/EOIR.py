import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LK_encoder(nn.Module):
    def __init__(self, in_cs, out_cs, kernel_size=5, stride=1, padding=2):
        super(LK_encoder, self).__init__()
        self.in_cs = in_cs
        self.out_cs = out_cs
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.regular = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, 3, 1, 1),
            nn.InstanceNorm3d(out_cs),
        )
        self.large = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_cs),
        )
        self.one = nn.Sequential(
            nn.Conv3d(in_cs, out_cs, 1, 1, 0),
            nn.InstanceNorm3d(out_cs),
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        x1 = self.regular(x)
        x2 = self.large(x)
        x3 = self.one(x)
        if self.in_cs == self.out_cs and self.stride == 1:
            x = x1 + x2 + x3 + x
        else:
            x = x1 + x2 + x3
        return self.prelu(x)

class encoder(nn.Module):
    def __init__(self, in_cs, out_cs, kernel_size=3, stride=1, padding=1):
        super(encoder, self).__init__()
        if kernel_size == 3:
            self.layer = nn.Sequential(
                nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding),
                nn.InstanceNorm3d(out_cs),
                nn.PReLU()
            )
        elif kernel_size > 3:
            self.layer = LK_encoder(in_cs, out_cs, kernel_size, stride, padding)

    def forward(self, x):
        return self.layer(x)
    
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

    def forward(self, src, flow):
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

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class dispWarp(nn.Module):

    def __init__(self, in_cs, ks=1, is_int=1):

        super(dispWarp, self).__init__()

        self.is_int = is_int

        self.disp_field_fea = nn.Sequential(
            nn.Conv3d(2*in_cs, 2*in_cs, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(2*in_cs, (ks*2+1)**3, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.get_flow = nn.Conv3d((ks*2+1)**3, 3, 3, 1, 1)
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def disp_field(self, x, y):

        feas = self.disp_field_fea(torch.cat((y+x, y-x), dim=1))
        flow = self.get_flow(feas)

        return flow

    def forward(self,x,y,transformer,up_flow,integrate):

        if up_flow is not None:
            x = transformer(x, up_flow)

        flow = self.disp_field(x, y)
        preint_flow = flow
        if self.is_int:
            flow = integrate(flow)

        if up_flow is not None:
            flow = flow + transformer(up_flow, flow)

        up_flow = self.up_tri(flow) * 2

        return preint_flow, flow, up_flow

class EOIR(nn.Module):

    def __init__(self, 
        img_size='(160, 224, 192)', 
        start_channel='16',
        lk_size= '5',
        cv_ks = '1',
        is_int = '1',
    ):
        super(EOIR, self).__init__()

        self.img_size = eval(img_size)
        self.start_channel = int(start_channel)
        self.lk_size = int(lk_size)
        self.cv_ks = int(cv_ks)
        self.is_int = int(is_int)

        N_s = self.start_channel
        self.simple_encoder = nn.Sequential(
            encoder(1,N_s,3,1,1),
            encoder(N_s,2*N_s,3,1,1),
            encoder(2*N_s,N_s,3,1,1),
        )

        ss = self.img_size
        self.transformers = nn.ModuleList([SpatialTransformer((ss[0]//2**i,ss[1]//2**i,ss[2]//2**i)) for i in range(5)])
        self.integrates = nn.ModuleList([VecInt((ss[0]//2**i,ss[1]//2**i,ss[2]//2**i), 7) for i in range(5)])
        self.disp_warp_4 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_3 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_2 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_1 = dispWarp(N_s, self.cv_ks, self.is_int)
        self.disp_warp_0 = dispWarp(N_s, self.cv_ks, self.is_int)

    def forward(self, x, y, registration=False):

        feas = self.simple_encoder(torch.cat([x, y], 0))
        x_feas, y_feas = torch.chunk(feas, 2, dim=0)

        x_0, y_0 = x_feas, y_feas

        x_1 = F.interpolate(x_0, scale_factor=0.5, mode='trilinear', align_corners=True)
        y_1 = F.interpolate(y_0, scale_factor=0.5, mode='trilinear', align_corners=True)

        x_2 = F.interpolate(x_1, scale_factor=0.5, mode='trilinear', align_corners=True)
        y_2 = F.interpolate(y_1, scale_factor=0.5, mode='trilinear', align_corners=True)

        x_3 = F.interpolate(x_2, scale_factor=0.5, mode='trilinear', align_corners=True)
        y_3 = F.interpolate(y_2, scale_factor=0.5, mode='trilinear', align_corners=True)

        x_4 = F.interpolate(x_3, scale_factor=0.5, mode='trilinear', align_corners=True)
        y_4 = F.interpolate(y_3, scale_factor=0.5, mode='trilinear', align_corners=True)

        int_flow_4, pos_flow_4, up_flow_4 = self.disp_warp_4(x_4,y_4,self.transformers[4],None,self.integrates[4])
        int_flow_3, pos_flow_3, up_flow_3 = self.disp_warp_3(x_3,y_3,self.transformers[3],up_flow_4,self.integrates[3])
        int_flow_2, pos_flow_2, up_flow_2 = self.disp_warp_2(x_2,y_2,self.transformers[2],up_flow_3,self.integrates[2])
        int_flow_1, pos_flow_1, up_flow_1 = self.disp_warp_1(x_1,y_1,self.transformers[1],up_flow_2,self.integrates[1])
        int_flow_0, pos_flow_0, _ = self.disp_warp_0(x_0,y_0,self.transformers[0],up_flow_1,self.integrates[0])

        int_flows = [int_flow_0, int_flow_1, int_flow_2, int_flow_3, int_flow_4]
        pos_flows = [pos_flow_0, pos_flow_1, pos_flow_2, pos_flow_3, pos_flow_4]

        if not registration:
            return int_flows, pos_flows
        else:
            return pos_flows[0]