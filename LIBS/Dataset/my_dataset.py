'''https://github.com/aleju/imgaug/issues/406
https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=8q8a2Ha9pnaz
'''

import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import albumentations as A

'''
transform_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Blur(blur_limit=0.4, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
    A.RandomRotate90(p=0.5),
    # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.9),
    A.Resize(image_shape[0], image_shape[1]),  # (height,weight),
    #     ToTensor()
    ], keypoint_params=A.KeypointParams(format='yx'))


class Dataset_CSV_alu(Dataset):
    def __init__(self, csv_file, transform=None, image_shape=None, test_mode=False):
        assert os.path.exists(csv_file), 'csv file does not exists'
        self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'

        self.image_shape = image_shape
        if transform is None:
            self.transform = A.Compose([
                A.Resize(image_shape[0], image_shape[1]),  # (height,weight)
                # ToTensor()
            ], keypoint_params=A.KeypointParams(format='yx'))
        else:
            self.transform = transform
        self.test_mode = test_mode

    def __getitem__(self, index):
        img_filename = self.df.iloc[index][0]
        assert os.path.exists(img_filename), 'image file does not exists'
        image = cv2.imread(img_filename)
        print(img_filename)
        center_x = self.df.iloc[index][1]
        center_y = self.df.iloc[index][2]

        keypoints = [
            (center_x, center_y)
        ]

        transformed = self.transform(image=image, keypoints=keypoints)
        transformed_image = transformed['image']
        transformed_keypoints = transformed['keypoints']
        center_x, center_y = transformed_keypoints[0][0], transformed_keypoints[0][1]
        x = transforms.ToTensor()(transformed_image)

        if not self.test_mode:
            y = np.array((center_x, center_y), dtype=np.float32)
            y = torch.from_numpy(y)

            return x, y
        else:
            return x

    def __len__(self):
        return len(self.df)

'''


class Dataset_CSV(Dataset):
    def __init__(self, csv_file, imgaug_iaa=None, image_shape=None, test_mode=False):
        assert os.path.exists(csv_file), 'csv file does not exists'
        self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'

        self.image_shape = image_shape
        if imgaug_iaa is not None:
            self.imgaug_iaa = imgaug_iaa
        else:
            self.imgaug_iaa = iaa.Sequential([
                iaa.Resize(size=(image_shape))  #(height, weight)
                ])

        self.test_mode = test_mode

    def __getitem__(self, index):
        img_filename = self.df.iloc[index][0]
        assert os.path.exists(img_filename), 'image file does not exists'
        image = cv2.imread(img_filename)

        center_x = self.df.iloc[index][1]
        center_y = self.df.iloc[index][2]

        #shape: height, width,  resize: width, height
        # if (self.image_shape is not None) and (image.shape[:2] != self.image_shape[:2]):
        #     center_x /= (image.shape[1] / self.image_shape[1])
        #     center_y /= (image.shape[0] / self.image_shape[0])
        #     image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))

        kps = KeypointsOnImage([
            Keypoint(x=center_x, y=center_y),
        ], shape=image.shape)

        image, kps_aug = self.imgaug_iaa(image=image, keypoints=kps)
        # before = kps.keypoints[0]
        after = kps_aug.keypoints[0]
        center_x, center_y = after.x, after.y

        # print(img_filename)
        # print(center_x, center_y)

        # (H,W,C)->(C,H,W) , normalization /255, -> Pytorch Tensor
        x = transforms.ToTensor()(image)

        if not self.test_mode:
            y = np.array((center_x, center_y), dtype=np.float32)
            y = torch.from_numpy(y)
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.df)


