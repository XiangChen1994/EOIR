from models.someWarpComplex import someWarpComplex
from models.hmorph_heart import hmorph_heart
from models.VxmDense import VxmDense
from models.VxmTransBrainComplex import VxmTransBrainComplex
from models.XMopherVxm import XMopherVxm
from models.VxmLKUnetComplex import VxmLKUnetComplex
from models.fourierNetAbdomenComplex import fourierNetAbdomenComplex
from models.VxmTransComplex import VxmTransComplex
from models.encoderOnlyComplex import encoderOnlyComplex
from models.encoderOnlyComplexLK import encoderOnlyComplexLK
def getModel(opt):

    model_name = opt['model']
    nkwargs = opt['nkwargs']
    model = None
    #print('333333333',model_name)

    if 'someWarpComplex' in model_name:
        model = someWarpComplex(start_channel=str(opt['start_channel']), is_int='1', lk_size='5', img_size=str(opt['img_size']))
    elif 'hmorph_heart' in model_name:
        model = hmorph_heart(**nkwargs)
    elif 'XMopherVxm' in model_name:
        model = XMopherVxm(n_channels=1)
    elif 'VxmDense' == model_name:
        model = VxmDense(inshape=opt['in_shape'],nb_unet_features=[opt['enc_nf'],opt['dec_nf']],bidir=0,int_steps=7,int_downsize=2)
    elif 'fourierNetAbdomenComplex' == model_name:
        model = fourierNetAbdomenComplex(in_channel=2,n_classes=3, start_channel='32',img_size=str(opt['in_shape']))
    elif 'VxmLKUnetComplex' == model_name:
        model = VxmLKUnetComplex(start_channel = '32',img_size=opt['in_shape'])
    elif 'VxmTransBrainComplex' in model_name:
        model = VxmTransBrainComplex(inshape=opt['in_shape'],nb_unet_features=[opt['enc_nf'],opt['dec_nf']],bidir=0,int_steps=7,int_downsize=2)
    elif 'VxmTransComplex' == model_name:
        model = VxmTransComplex(inshape=opt['in_shape'],nb_unet_features=[opt['enc_nf'],opt['dec_nf']],bidir=0,int_steps=7,int_downsize=2)
    elif 'encoderOnlyComplex' == model_name:
        model = encoderOnlyComplex(img_size=str(opt['img_size']), start_channel=str(opt['start_channel']), lk_size='5', cv_ks='1', is_int='1')
    elif 'encoderOnlyComplexLK' == model_name:
        model = encoderOnlyComplexLK(img_size=str(opt['img_size']), start_channel=str(opt['start_channel']), lk_size='5', cv_ks='1', is_int='1')
    else:
        raise ValueError("Model %s not recognized." % model_name)

    model = model.cuda()
    print("----->>>> Model %s is built ..." % model_name)

    return model