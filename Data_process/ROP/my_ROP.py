
import cv2
import os
import math
import json
import csv
from LIBS.ImgPreprocess.my_image_helper import resize_images_dir

do_preprocess = False

dir_original = '/media/ubuntu/data2/Fover_center/ROP/20200912/original/'
preprocess_image_size = 384
dir_preprocess ='/media/ubuntu/data2/Fover_center/ROP/20200912/preprocess384/'
dir_tmp = '/tmp2/ROP/'

if do_preprocess:
    resize_images_dir(dir_original, dir_preprocess,
                  convert_image_to_square=True, image_size=preprocess_image_size)

filename_csv = os.path.join(os.path.abspath('.'), 'ROP.csv')
if os.path.exists(filename_csv):
    os.remove(filename_csv)

with open(filename_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['images', 'x', 'y'])

    for dir_path, subpaths, files in os.walk(dir_original, False):
        for f in files:
            full_filename = os.path.join(dir_path, f)
            file_base, file_ext = os.path.splitext(full_filename)
            if file_ext.lower() not in ['.json']:
                continue

            img_file_original = file_base + '.jpg'
            assert os.path.exists(img_file_original), f'{img_file_original} image file not exist!'
            print(full_filename)
            img_file_preprocess = img_file_original.replace(dir_original, dir_preprocess)
            assert os.path.exists(img_file_preprocess), f'{img_file_original} processed image file not exist!'

            with open(full_filename, 'r') as json_file:
                data = json.load(json_file)
                (x, y) = data['shapes'][0]['points'][0]
                img1 = cv2.imread(img_file_original)
                (height, width) = img1.shape[0:2]

                if width > height:
                    offset = math.floor((width - height) / 2)
                    y += offset
                else:
                    offset = math.floor((height - width) / 2)
                    x += offset

                rescale_ratio = max(height, width) / preprocess_image_size
                x /= rescale_ratio
                y /= rescale_ratio

                csv_writer.writerow([img_file_preprocess, round(x, 2), round(y, 2)])

                if 'dir_tmp' in dir():
                    img_preprocess = cv2.imread(img_file_preprocess)
                    (height, width) = img_preprocess.shape[0:2]
                    thickness = int(max(img_preprocess.shape[:-1]) / 100)
                    img_draw_circle = cv2.circle(img_preprocess, (int(x), int(y)), thickness, (0, 0, 255), -1)

                    image_file_tmp = img_file_preprocess.replace(dir_preprocess, dir_tmp)
                    os.makedirs(os.path.dirname(image_file_tmp), exist_ok=True)
                    cv2.imwrite(image_file_tmp, img_draw_circle)
                    print(image_file_tmp)


print('OK!')




