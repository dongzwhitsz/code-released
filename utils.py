from model.model_densenet import ModelDensenet
from model.resnet import Resnet
from model.model_plain import ModelPlain
from model.model_dongzw import Dongzw


def get_model(model_name,purpose=0):
    if model_name == 'Dongzw':
        model = Dongzw(purpose)
    else:
        raise ValueError('specify the model')
    return model
