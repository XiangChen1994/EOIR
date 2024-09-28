import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from models.backbones.xmopher.net import Head
from models.backbones.xmopher.stn import SpatialTransformer, Re_SpatialTransformer


class XMopherVxm(nn.Module):

    #@store_config_args
    def __init__(
        self,
        n_channels=1
        ):
        super().__init__()

        self.model = Head(n_channels=n_channels)

    def forward(self, source, target, x_seg=None, y_seg=None, registration=False, is_erf=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''
        f_xy, f_yx, seg_xy, seg_yx, flow = self.model(source, target)

        #return f_xy, f_yx
        return f_xy, flow
