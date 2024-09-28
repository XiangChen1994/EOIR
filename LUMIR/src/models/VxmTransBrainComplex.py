import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from models.backbones.voxelmorph import default_unet_features
from models.backbones.voxelmorph.torch import layers
from models.backbones.voxelmorph.torch.networks import Unet
from models.backbones.voxelmorph.torch.modelio import LoadableModel, store_config_args

from models.backbones.transmorph.transMorphBrain import CONFIGS, TransMorph

class VxmTransBrainComplex(LoadableModel):

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 trans_type='n',
                 integrate='1'):

        super().__init__()

        self.training = True
        self.trans_type = trans_type
        self.integrate = int(integrate)

        print("trans_type: %s, integrate: %d" % (self.trans_type, self.integrate))

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if self.trans_type == 'l':
            self.model = TransMorph(CONFIGS['TransMorph-Large'])
        elif self.trans_type == 's':
            self.model = TransMorph(CONFIGS['TransMorph-Small'])
        elif self.trans_type == 't':
            self.model = TransMorph(CONFIGS['TransMorph-Tiny'])
        elif self.trans_type == 'n':
            self.model = TransMorph(CONFIGS['TransMorph'])
            # fp = 'TransMorph_Validation_dsc0.857.pth'
            # best_model = torch.load('./../../../data/oasis_pkl/pths/'+fp)['state_dict']
            # self.model.load_state_dict(best_model)
            # print("Model loaded from: %s" % fp)

        self.resize = layers.ResizeTransform(int_downsize, ndims)
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)

        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, x_seg=None, y_seg=None, registration=False, is_erf=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        x = torch.cat([source, target], dim=1)
        y_source, pos_flow = self.model(x)

        if not registration:
            return y_source, pos_flow
        else:
            return y_source, pos_flow