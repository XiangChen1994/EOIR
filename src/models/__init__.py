from models.EOIR import EOIR
from models.EOIR_ACDC import EOIR_ACDC
from models.EOIR_OASIS import EOIR_OASIS

def getModel(opt):

    model_name = opt['model']
    nkwargs = opt['nkwargs']
    model = None

    if 'EOIR_ACDC' in model_name:
        model = EOIR_ACDC(**nkwargs)
    if 'EOIR_OASIS' in model_name:
        model = EOIR_OASIS(**nkwargs)
    if 'EOIR' in model_name:
        model = EOIR(**nkwargs)
    else:
        raise ValueError("Model %s not recognized." % model_name)

    model = model.cuda()
    print("----->>>> Model %s is built ..." % model_name)

    return model