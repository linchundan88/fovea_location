import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
import pretrainedmodels


num_regression = 2  #fovea, x,y
model_name = 'xception'  # 'xception', 'inceptionresnetv2', 'inceptionv3'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
num_filters = model.last_linear.in_features
model.last_linear = nn.Linear(num_filters, num_regression)
model_file = os.path.join(os.path.abspath('..'), 'train_models', 'ROP', 'xception_epoch14.pth')
state_dict = torch.load(model_file, map_location='cpu')
model.load_state_dict(state_dict)

image_file = '/media/ubuntu/data2/Fover_center/DR0_4_inconsistent/512/original/DR4/66b17a08-a74d-11e8-94f6-6045cb817f5b.jpg'
# /media/ubuntu/data2/Fover_center/DR0_4_inconsistent/512/original/DR4/66b17a08-a74d-11e8-94f6-6045cb817f5b.jpg,97.0,298.0
import cv2
image = cv2.imread(image_file)
image_shape = (299, 299)

from LIBS.NeuralNetworks.Helper.my_predict import predict_one_image
y = predict_one_image(model, image, image_shape=image_shape)

(center_x, center_y) = y
center_x_originl = center_x * (image.shape[1] / image_shape[1])
center_y_originl = center_y * (image.shape[0] / image_shape[0])


thickness = int(max(image.shape[:-1]) / 100)
img_draw_circle = cv2.circle(image, (int(center_x_originl), int(center_y_originl)), thickness, (0, 0, 255), -1)
cv2.imwrite('/tmp2/aaa.jpg', img_draw_circle)

print('OK')


