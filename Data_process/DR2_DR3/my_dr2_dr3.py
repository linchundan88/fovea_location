import os
import json
import csv
import cv2
import math

dir_original = '/media/ubuntu/data2/Fover_center/DR2_DR3_500/original/'
dir_preprocess = '/media/ubuntu/data2/Fover_center/DR2_DR3_500/preprocess512/'
dir_tmp = '/tmp2/DR2_DR3/'

do_preprocess = True
preprocess_image_size = 512
from LIBS.ImgPreprocess.my_image_helper import resize_images_dir
if do_preprocess:
    resize_images_dir(dir_original, dir_preprocess,
                  convert_image_to_square=True, image_size=preprocess_image_size)

filename_csv = os.path.join(os.path.abspath('.'), 'DR2_DR3.csv')
if os.path.exists(filename_csv):
    os.remove(filename_csv)

with open(filename_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['images', 'x', 'y'])

    for file_type in ['DR2.json', 'DR3.json']:
        filename_json = os.path.join(os.path.abspath('.'), file_type)

        with open(filename_json, 'r') as f:
            dict1 = json.load(f)

            for k, v in dict1.items():
                if '0' not in v['regions']:
                    continue

                image_file_original = os.path.join(dir_original, v['filename'])
                assert os.path.exists(image_file_original), f'file: {image_file_original} not exists.'
                x = v['regions']['0']['shape_attributes']['cx']
                y = v['regions']['0']['shape_attributes']['cy']
                # r = v['regions']['0']['shape_attributes']['r']

                img1 = cv2.imread(image_file_original)
                (height, width) = img1.shape[0:2]

                if width > height:
                    offset = math.ceil((width - height) / 2)
                    y += offset
                elif height > width:
                    offset = math.ceil((height - width) / 2)
                    x += offset

                rescale_ratio = max(height, width) / preprocess_image_size
                x /= rescale_ratio
                y /= rescale_ratio

                image_file_preprocess = image_file_original.replace(dir_original, dir_preprocess)
                img_preprocess = cv2.imread(image_file_preprocess)

                csv_writer.writerow([image_file_preprocess, round(x, 2), round(y, 2)])

                if 'dir_tmp' in dir():
                    (height, width) = img_preprocess.shape[0:2]
                    thickness = int(max(img_preprocess.shape[:-1]) / 100)
                    img_draw_circle = cv2.circle(img_preprocess, (int(x), int(y)), thickness, (0, 0, 255), -1)

                    image_file_tmp = image_file_original.replace(dir_original, dir_tmp)
                    os.makedirs(os.path.dirname(image_file_tmp), exist_ok=True)
                    cv2.imwrite(image_file_tmp, img_draw_circle)
                    print(image_file_tmp)
print('OK')