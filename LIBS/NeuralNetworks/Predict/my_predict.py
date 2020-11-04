
import os
import numpy as np

import torch
from torch.nn import functional as F
import torch.nn as nn
from LIBS.NeuralNetworks.Predict.my_img_to_tensor import preprocess_image
from torch.utils.data import DataLoader
from LIBS.Dataset.my_dataset import Dataset_CSV
from LIBS.NeuralNetworks.Helper.my_is_inception import is_inception_model


def predict_csv(model, filename_csv, image_shape,
                model_convert_gpu=True,
               batch_size=64,
               argmax=True):

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
    list_probs = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(data_loader):
            print('batch:', batch_idx)
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1).data
            list_probs.append(probabilities.cpu().numpy())

    probs = np.vstack(list_probs)

    if argmax:
        y_preds = probs.argmax(axis=-1)
        return probs, y_preds
    else:
        return probs


def predict_csv_ensemble(model_dicts, filename_csv,
                         model_convert_gpu=True,
                         dimension='2D', batch_size=64,
                         argmax=True):

    list_probs = []
    for index, model_dict in enumerate(model_dicts):
        probs = predict_csv(model_dict['model'], filename_csv, model_dict['image_shape'],
            model_convert_gpu, dimension, batch_size, argmax=False)

        list_probs.append(probs * model_dict['model_weight'])

    probs_total = np.stack(list_probs, axis=0)
    probs_average = np.average(probs_total, axis=0)

    if argmax:
        y_preds = probs_average.argmax(axis=-1)
        return probs_average, y_preds
    else:
        return probs_average


def predict_one_image(model, img_file, model_convert_gpu=True,
                      image_shape=(299, 299), argmax=False):
    assert os.path.exists(img_file), 'fine not found!'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_convert_gpu and torch.cuda.device_count() > 0:
        model.to(device)  # model.cuda()
    model.eval()

    inputs = preprocess_image(img_file, image_shape)
    inputs = inputs.to(device)
    logits = model(inputs)
    probs = F.softmax(logits, dim=1).data.squeeze(dim=0)  #batch contains one sample
    # probabilities = torch.softmax(outputs, dim=0)

    if argmax:
        preds = torch.argmax(probs)
        return probs.cpu().numpy(), preds.cpu().numpy()
    else:
        return probs.cpu().numpy()


def predict_one_image_ensemble(model_dicts, img_file, model_convert_gpu=True,
                               argmax=False):
    list_probs = []
    for index, model_dict in enumerate(model_dicts):
        probs = predict_one_image(model_dict['model'], img_file,
                                  model_convert_gpu=model_convert_gpu,
                                  image_shape= model_dict['image_shape'],
             argmax=False)
        list_probs.append(probs * model_dict['model_weight'])

    probs_total = np.stack(list_probs, axis=0)
    probs_average = np.average(probs_total, axis=0)

    if argmax:
        y_preds = probs_average.argmax(axis=-1)
        return probs_average, y_preds
    else:
        return probs_average

