'''
get_position:  according to thretholds find the area of retina, return (bottom, top, left, right)

crop_image: using (bottom, top, left, right)

create_img_fover: create fovea based on Fovera coordinates
'''


import pandas as pd
import os, cv2
import numpy as np


from LIBS.ImgPreprocess.my_image_helper import crop_image, resize_images_dir
from LIBS.ImgPreprocess.my_preprocess import get_fundus_border


def create_img_fover(img1, x, y):
    if isinstance(img1, str):
        img1 = cv2.imread(img1)

    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    img2 = np.zeros(img1.shape)

    img3 = cv2.circle(img2, (x, y), 100, (255, 255, 255), -1)

    img4 = img3.astype(np.uint8)

    return img4

def create_img_fover_dir():
    df = pd.read_csv('IDRiD_Fovea_Center.csv')

    fover_dir = '/home/ubuntu/Fover_center/IDRID/C. Localization/1. Original Images/'
    if not fover_dir.endswith('/'):
        fover_dir = fover_dir + '/'

    dest_dir = '/tmp2/'
    if not dest_dir.endswith('/'):
        dest_dir = dest_dir + '/'

    count = len(df.index)
    for i in range(count):
        #read original fover image, based on it size and coordinates, create fover image
        filename_fover = os.path.join(fover_dir, df.at[i, 'Image']+'.jpg')

        center_x = df.at[i, 'X-Coordinate']
        center_y = df.at[i, 'Y-Coordinate']

        img_fover = create_img_fover(filename_fover, center_x, center_y)

        filename_dest = filename_fover.replace(fover_dir, dest_dir)

        if not os.path.exists(os.path.dirname(filename_dest)):
            os.makedirs(os.path.dirname(filename_dest))

        print(filename_dest)
        cv2.imwrite(filename_dest, img_fover)


resize_images_dir('/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/512', imgsize=512)

exit(0)

base_dir = '/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/preprocess/original'

for dir_path, subpaths, files in os.walk(base_dir, False):
    for f in files:
        img_file_source = os.path.join(dir_path, f)

        filename, file_extension = os.path.splitext(img_file_source)

        if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
            print('file ext name:', f)
            continue

        img1 = cv2.imread(img_file_source)

        bottom, top, left, right = get_fundus_border(img1)

        img2 = crop_image(img_file_source, bottom, top, left, right)
        print(img_file_source)
        cv2.imwrite(img_file_source, img2)

        img_file_fovea = img_file_source.replace('/original/', '/fovea/')

        img3 = crop_image(img_file_fovea, bottom, top, left, right)
        print(img_file_fovea)
        cv2.imwrite(img_file_fovea, img3)

exit(0)

resize_images_dir('/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/512', imgsize=512)

print('OK')
