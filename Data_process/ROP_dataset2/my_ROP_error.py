
import cv2
import os
import math
import json
import csv
from LIBS.ImgPreprocess.my_image_helper import resize_images_dir
import shutil

do_preprocess = False

dir_original = '/media/ubuntu/data2/Fover_center/ROP/2020_11_4/original/'
preprocess_image_size = 384
dir_preprocess ='/media/ubuntu/data2/Fover_center/ROP/2020_11_4/preprocess384/'
dir_err = '/tmp2/ROP_error2020_11_4/'

if do_preprocess:
    resize_images_dir(dir_original, dir_preprocess,
                  convert_image_to_square=True, image_size=preprocess_image_size)


if True:
    for dir_path, subpaths, files in os.walk(dir_original, False):
        for f in files:
            full_filename = os.path.join(dir_path, f)
            file_base, file_ext = os.path.splitext(full_filename)
            if file_ext.lower() not in ['.json']:
                continue

            img_file_original = file_base + '.jpg'
            assert os.path.exists(img_file_original), f'{img_file_original} image file not exist!'
            # print(full_filename)
            img_file_preprocess = img_file_original.replace(dir_original, dir_preprocess)
            assert os.path.exists(img_file_preprocess), f'{img_file_original} processed image file not exist!'

            with open(full_filename, 'r') as json_file:
                data = json.load(json_file)
                if len(data['shapes']) > 1:
                    print(full_filename)

                    file_error = full_filename.replace(dir_original, dir_err)
                    shutil.copy(full_filename, file_error)

                    file_error = file_error.replace('.json', '.jpg')
                    shutil.copy(full_filename.replace('.json', '.jpg'), file_error)

                    print(file_error)


print('OK!')




