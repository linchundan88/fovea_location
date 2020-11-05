
import os
import numpy as np

import torch
import torch.nn as nn
from LIBS.NeuralNetworks.Helper.my_img_to_tensor import img_to_tensor
from torch.utils.data import DataLoader
from LIBS.Dataset.my_dataset import Dataset_CSV
from LIBS.NeuralNetworks.Helper.my_is_inception import is_inception_model


def predict_csv(model, filename_csv, image_shape,
                model_convert_gpu=True,
               batch_size=64):

    assert os.path.exists(filename_csv), "csv file not exist"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_convert_gpu and torch.cuda.device_count() > 0:
        model.to(device)
    is_inception = is_inception_model(model)
    if model_convert_gpu and torch.cuda.device_count() > 1 and (not is_inception):
        model = nn.DataParallel(model)
    model.eval()

    dataset = Dataset_CSV(csv_file=filename_csv, image_shape=image_shape, test_mode=True)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             num_workers=4)
    list_preds = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(data_loader):
            print('batch:', batch_idx)
            inputs = inputs.to(device)
            outputs = model(inputs)
            list_preds.append(outputs.cpu().numpy())

    preds = np.concatenate(list_preds, axis=0)

    return preds



def predict_one_image(model, img_file, model_convert_gpu=True,
                      image_shape=(299, 299)):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_convert_gpu and torch.cuda.device_count() > 0:
        model.to(device)
    model.eval()

    inputs = img_to_tensor(img_file, image_shape)
    inputs = inputs.to(device)

    with torch.no_grad():
        preds = model(inputs)[0]

    return preds.cpu().numpy()

