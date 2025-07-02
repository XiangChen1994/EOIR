from models.EOIR import encoderOnlyComplex

def getModel(opt):

    model_name = opt['model']
    nkwargs = opt['nkwargs']
    model = None
    #print('333333333',model_name)

    if 'EOIR' == model_name:
        model = encoderOnlyComplex(img_size=str(opt['img_size']), start_channel=str(opt['start_channel']), lk_size='5', cv_ks='1', is_int='1')
    else:
        raise ValueError("Model %s not recognized." % model_name)

    model = model.cuda()
    print("----->>>> Model %s is built ..." % model_name)

    return model
