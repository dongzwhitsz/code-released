from model.model_dongzw import Dongzw
from model.model_dongzw_v2 import Dongzw_V2
from model.model_dongzw_v3 import Dongzw_V3


def get_model(model_name,purpose=0):
    if model_name == 'Dongzw':
        model = Dongzw(purpose)
    elif model_name == 'Dongzw_V2':
        model = Dongzw_V2(purpose)
    elif model_name == 'Dongzw_V3':
        model = Dongzw_V3(purpose)
    else:
        raise ValueError('specify the model')
    return model
