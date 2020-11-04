import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as optim_plus
# from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from LIBS.Dataset.my_dataset import Dataset_CSV
from LIBS.NeuralNetworks.my_train import train_regression
from imgaug import augmenters as iaa
import albumentations as A
import pretrainedmodels


#region prepare dataset

image_shape = (299, 299)

train_type = 'fovea_adults_ROP'
data_version = 'V3'
filename_csv_train = os.path.join(os.path.abspath('../'),
               'datafiles', data_version,  f'{train_type}_train.csv')
filename_csv_valid = os.path.join(os.path.abspath('../'),
               'datafiles', data_version,  f'{train_type}_valid.csv')
filename_csv_test = os.path.join(os.path.abspath('../'),
               'datafiles', data_version,  f'{train_type}_test.csv')


imgaug_iaa = iaa.Sequential([
    # iaa.CropAndPad(percent=(-0.04, 0.04)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),

    iaa.GaussianBlur(sigma=(0.0, 0.5)),
    iaa.MultiplyBrightness(mul=(0.8, 1.2)),
    iaa.contrast.LinearContrast((0.8, 1.2)),
    iaa.Sometimes(0.9, iaa.Add((-8, 8))),
    iaa.Sometimes(0.9, iaa.Affine(
        scale=(0.98, 1.02),
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-10, 10))),
    iaa.Resize(size=(image_shape))  #(height, weight)
])

batch_size_train, batch_size_valid = 32, 64

num_workers = 6
ds_train = Dataset_CSV(csv_file=filename_csv_train, imgaug_iaa=imgaug_iaa, image_shape=image_shape)
loader_train = DataLoader(ds_train, batch_size=batch_size_train, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
ds_valid = Dataset_CSV(csv_file=filename_csv_valid, image_shape=image_shape)
loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid,
                          num_workers=num_workers, pin_memory=True)
ds_test = Dataset_CSV(csv_file=filename_csv_test, image_shape=image_shape)
loader_test = DataLoader(ds_test, batch_size=batch_size_valid,
                         num_workers=num_workers, pin_memory=True)

#endregion

#region training
num_regression = 2  #fovea, x,y
save_model_dir = '/tmp2/fovea_location_models_2020_11_4_3'
train_times = 2
# 'xception', 'inceptionresnetv2', 'inceptionv3'
for i in range(train_times):
    for model_name in ['xception', 'inceptionresnetv2', 'inceptionv3', 'nasnetamobile']:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        num_filters = model.last_linear.in_features
        model.last_linear = nn.Linear(num_filters, num_regression)

        criterion = nn.MSELoss()  # criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        # optimizer = optim_plus.radam(model.parameters(), lr=0.001, weight_decay=0)
        # optimizer = optim_plus.Lookahead(optimizer, k=5, alpha=0.5)

        epochs_num = 20
        scheduler = StepLR(optimizer, step_size=3, gamma=0.3)
        # scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
        # scheduler = GradualWarmupScheduler(optimizer, multiplier=4, total_epoch=2, after_scheduler=scheduler)

        train_regression(model,
              loader_train=loader_train, loader_valid=loader_valid, loader_test=loader_test,
              criterion=criterion, optimizer=optimizer,
              scheduler=scheduler,
              epochs_num=epochs_num, log_interval_train=10, log_interval_test=10,
              show_detail_train=False, show_detail_test=False,
              save_model_dir=os.path.join(save_model_dir, train_type, data_version, model_name, str(i))
              )

#endregion

print('OK')

