import os
import csv
import cv2
import pandas as pd
import math
from LIBS.ImgPreprocess.my_image_helper import image_to_square, get_position, crop_image


dir_original = '/media/ubuntu/data2/Fover_center/IDRID/original'
dir_tmp = '/tmp2/IDRID'
dir_preprocess = '/media/ubuntu/data2/Fover_center/IDRID/preprocess512'
preprocess_image_size = 512

filename_csv = os.path.join(os.path.abspath('.'), 'IDRID.csv')
if os.path.exists(filename_csv):
    os.remove(filename_csv)

with open(filename_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['images', 'x', 'y'])
    for file_type in ['IDRiD_Fovea_Center_Training Set_Markups.csv',
                      'IDRiD_Fovea_Center_Testing Set_Markups.csv']:
        filename_csv = os.path.join(os.path.abspath('.'), file_type)
        df = pd.read_csv(filename_csv)
        for _, row in df.iterrows():
            if row[0] is None:
                continue
            if not isinstance(row[0], str) and math.isnan(row[0]):
                continue

            if 'Training' in file_type:
                image_file_original = os.path.join(dir_original, 'TrainingSet',
                                                   row[0] + '.jpg')
            if 'Testing' in file_type:
                image_file_original = os.path.join(dir_original, 'TestingSet',
                                                   row[0] + '.jpg')
            x, y = row[1], row[2]

            image_original = cv2.imread(image_file_original)

            (height, width) = image_original.shape[:-1]
            (bottom, top, left, right) = get_position(image_file_original)
            img_preprocess = crop_image(image_original, bottom, top, left, right)
            x, y = x - left, y - bottom

            (height, width) = img_preprocess.shape[:-1]
            img_preprocess = image_to_square(img_preprocess, image_size=preprocess_image_size)
            if width > height:
                y += math.ceil((width - height) / 2)
            elif height > width:
                x += math.ceil((height - width) / 2)
            rescale_ratio = max(width, height) / preprocess_image_size
            x /= rescale_ratio
            y /= rescale_ratio

            image_file_preprocess = image_file_original.replace(dir_original, dir_preprocess)
            os.makedirs(os.path.dirname(image_file_preprocess), exist_ok=True)
            cv2.imwrite(image_file_preprocess, img_preprocess)
            csv_writer.writerow([image_file_preprocess, round(x, 2), round(y, 2)])
            print(image_file_preprocess)

            if 'dir_tmp' in dir():
                (height, width) = img_preprocess.shape[0:2]
                thickness = int(max(img_preprocess.shape[:-1]) / 100)
                img_draw_circle = cv2.circle(img_preprocess, (int(x), int(y)), thickness, (0, 0, 255), -1)

                image_file_tmp = image_file_original.replace(dir_original, dir_tmp)
                os.makedirs(os.path.dirname(image_file_tmp), exist_ok=True)
                cv2.imwrite(image_file_tmp, img_draw_circle)
                print(image_file_tmp)

print('OK')