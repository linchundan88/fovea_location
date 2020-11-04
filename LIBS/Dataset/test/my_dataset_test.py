
from LIBS.Dataset.my_dataset import Dataset_CSV
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import uuid

filename_csv = 'fovea.csv'
image_shape = (299, 299)

dir_tmp = '/tmp2/dataset_test/'

iaa = iaa.Sequential([
    # iaa.CropAndPad(percent=(-0.04, 0.04)),
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Flipud(0.2),  # horizontally flip 50% of the images

    iaa.GaussianBlur(sigma=(0.0, 0.5)),
    iaa.MultiplyBrightness(mul=(0.8, 1.2)),
    iaa.contrast.LinearContrast((0.8, 1.2)),
    iaa.Sometimes(0.9, iaa.Add((-8, 8))),
    iaa.Sometimes(0.9, iaa.Affine(
        scale=(0.98, 1.02),
        translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
        rotate=(-15, 15),
    )),
])

iaa = None

batch_size = 32

dataset= Dataset_CSV(csv_file=filename_csv, imgaug_iaa=iaa, image_shape=image_shape)
loader = DataLoader(dataset, batch_size=batch_size,
                          num_workers=6)

for x, y in loader:
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    for i in range(x.shape[0]):
        img = x[i]
        img = np.transpose(img, (1, 2, 0))   #(C,H,W) -> (H,W,C)
        img *= 255

        r_x = y[i][0]
        r_y = y[i][1]

        (height, width) = img.shape[0:2]
        thickness = int(max(img.shape[:-1]) / 100)

        # img1 = img.astype(np.uint8)
        img1 = img.copy() #avoid typeError: Expected Ptr<cv::UMat> for argument 'img'
        img_draw_circle = cv2.circle(img1,
                                     (int(r_x), int(r_y)), thickness, (0, 0, 255), -1)

        file_dest = os.path.join(dir_tmp, str(uuid.uuid4())+'.jpg')
        os.makedirs(os.path.dirname(file_dest), exist_ok=True)
        cv2.imwrite(file_dest, img_draw_circle)

        print(file_dest)

print('OK')