'''
  the difference between valid() and test() is that valid compute loss.
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from LIBS.NeuralNetworks.Predict.my_predict import predict_one_image, predict_one_image_ensemble,\
    predict_csv

def test_image():
    img_file = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384/R0/2019_2_28/GLEYY01-S522rec522-07-9669652d-9271-423a-a0cb-02521e055ff9.jpg'
    # img_file = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384/R3s/清晰可见/LZRCH01-S576rec576-05-e5ae4e37-3ec2-4ef2-ac84-1403406ec833.jpg'
    # img_file = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384/R2/清晰可见/680066e9-a74d-11e8-94f6-6045cb817f5b.jpg'
    import pretrainedmodels
    num_class = 2
    model_name = 'xception'
    model = pretrainedmodels.__dict__[model_name](num_classes=num_class, pretrained=None)

    state_dict = torch.load('/tmp7/backup/epoch4.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    #probabilities.ndim = 1 , predict_class.ndim=0
    probabilities, predict_class = predict_one_image(model, img_file, image_shape=(299, 299), argmax=True)

    print(probabilities, predict_class)

def test_image_ensemble():
    img_file = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384/R0/2019_2_28/GLEYY01-S522rec522-07-9669652d-9271-423a-a0cb-02521e055ff9.jpg'
    # img_file = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384/R3s/清晰可见/LZRCH01-S576rec576-05-e5ae4e37-3ec2-4ef2-ac84-1403406ec833.jpg'
    # img_file = '/media/ubuntu/data1/糖网项目/DR分级英国标准_20190119_无杂病/DR/preprocess384/R2/清晰可见/680066e9-a74d-11e8-94f6-6045cb817f5b.jpg'
    import pretrainedmodels
    num_class = 2
    model_name = 'xception'
    model = pretrainedmodels.__dict__[model_name](num_classes=num_class, pretrained=None)

    image_shape = (299, 299)

    state_dict = torch.load('/tmp7/backup/epoch4.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    model_dicts = []
    model_dict1 = {'model': model, 'image_shape':image_shape, 'model_weight': 1}
    model_dicts.append(model_dict1)
    model_dict1 = {'model': model, 'image_shape':image_shape, 'model_weight': 1}
    model_dicts.append(model_dict1)

    #probabilities.ndim = 1 , predict_class.ndim=0
    probabilities, predict_class = predict_one_image_ensemble(model_dicts, img_file, argmax=True)

    print(probabilities, predict_class)


def test_csv():
    import pretrainedmodels
    num_class = 2
    model_name = 'xception'
    model = pretrainedmodels.__dict__[model_name](num_classes=num_class, pretrained=None)
    state_dict = torch.load('/tmp7/backup/epoch4.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    data_version = 'v1'
    filename_csv = os.path.join(os.path.abspath('../../../'), 'DR',
                        'datafiles', 'DR_english', 'split_patid_test_{}.csv'.format(data_version))

    probs, predicts = predict_csv(model, filename_csv, image_shape=(299, 299), argmax=True)
    print(probs, predicts)

if __name__ == '__main__':
    test_image()
    # test_csv()

    # test_image_ensemble()

    print('ok')


