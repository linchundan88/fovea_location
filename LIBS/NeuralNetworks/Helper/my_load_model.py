
import os

import torch
import torch.nn as nn
from LIBS.NeuralNetworks.Helper.my_is_inception import is_inception_model
import pretrainedmodels

def load_model(model_name, model_file, num_class=2):
    if not os.path.exists(model_file):
        raise Exception(model_file + ' not found!')

    if model_name in ['xception', 'inceptionresnetv2', 'inceptionv3']:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        num_filters = model.last_linear.in_features
        model.last_linear = nn.Linear(num_filters, num_class)
    else:
        raise Exception('model_name error')

    state_dict = torch.load('model_file', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    is_inception = is_inception_model(model)
    if is_inception:
        model.AuxLogits.training = False
        num_filters = model.AuxLogits.fc.in_features
        num_class = model.last_linear.out_features
        model.AuxLogits.fc = nn.Linear(num_filters, num_class)

    if torch.cuda.device_count() > 1 and (not is_inception):
        model = nn.DataParallel(model)
    if torch.cuda.device_count() > 0:
        model.cuda()

    return model