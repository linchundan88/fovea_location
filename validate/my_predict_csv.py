import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import pretrainedmodels
import pandas as pd
import cv2
import shutil

num_regression = 2  #fovea, x,y
model_name = 'xception'  # 'xception', 'inceptionresnetv2', 'inceptionv3'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
num_filters = model.last_linear.in_features
model.last_linear = nn.Linear(num_filters, num_regression)
model_file = os.path.join(os.path.abspath('..'), 'train_models', 'ROP', 'xception_epoch14.pth')
state_dict = torch.load(model_file, map_location='cpu')
model.load_state_dict(state_dict)

train_type = 'fovea_ROP_patient_based'
data_version = 'V3'
filename_csv = os.path.join(os.path.abspath('../'),
               'datafiles', data_version,  f'{train_type}.csv')


from LIBS.NeuralNetworks.Helper.my_predict import predict_csv
image_shape = (299, 299)
results = predict_csv(model, filename_csv, image_shape,
                model_convert_gpu=True, batch_size=64)

# np.save('preds.npy', results)
df = pd.read_csv(filename_csv)

threshold = 15

dir_error = '/tmp2/error'
for index, row in df.iterrows():
    file_img = row['images']

    image1 = cv2.imread(file_img)
    x_gt, y_gt = row['x'], row['y']
    x_pred, y_pred = results[index]
    x_pred = x_pred / image_shape[1] * image1.shape[1]
    y_pred = y_pred / image_shape[0] * image1.shape[0]

    threshold1 = threshold / image_shape[1] * image1.shape[1]

    if abs(x_pred-x_gt) > threshold or abs(y_pred-y_gt) > threshold1:
        print(f'{x_gt}, {y_gt}, {x_pred}, {y_pred}')

        file_img = file_img.replace('/preprocess384/', '/original/')
        _, filename = os.path.split(file_img)
        file_base, file_ext = os.path.splitext(filename)

        file_dest = os.path.join(dir_error, f'{file_base}{file_ext}')
        os.makedirs(os.path.dirname(file_dest), exist_ok=True)
        shutil.copy(file_img, file_dest)

        thickness = int(max(image1.shape[:-1]) / 100)
        img_draw_circle = cv2.circle(image1, (int(x_pred), int(y_pred)), thickness, (0, 0, 255), -1)
        file_dest_draw = os.path.join(dir_error, f'{file_base }_point{file_ext}')
        cv2.imwrite(file_dest_draw, img_draw_circle)


print('OK')



