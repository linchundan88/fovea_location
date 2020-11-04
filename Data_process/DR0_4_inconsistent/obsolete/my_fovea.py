import json, os, csv, cv2
import pandas as pd
import numpy as np


def create_img_fover(img1, x, y, r):
    if isinstance(img1, str):
        img1 = cv2.imread(img1)

    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    img2 = np.zeros(img1.shape)

    img3 = cv2.circle(img2, (x, y), r, (255, 255, 255), -1)

    img4 = img3.astype(np.uint8)

    return img4

def op_json(file_json, base_dir = '', dest_dir = '',  dr_type='DR2'):
    if not base_dir.endswith('/'):
        base_dir = base_dir + '/'
    if not dest_dir.endswith('/'):
        dest_dir = dest_dir + '/'

    with open(file_json, 'r', encoding='utf-8') as load_f:
        dict1 = json.load(load_f)

        for k, v in dict1.items():
            filename_orig = os.path.join(base_dir, dr_type, v['filename'] )

            if not '0' in v['regions']:
                os.remove(filename_orig)
                continue

            x = v['regions']['0']['shape_attributes']['cx']
            y = v['regions']['0']['shape_attributes']['cy']
            r = v['regions']['0']['shape_attributes']['r']

            img1 = create_img_fover(filename_orig, x, y, r)

            filename_dest = filename_orig.replace(base_dir, dest_dir)

            if not os.path.exists(os.path.dirname(filename_dest)):
                os.makedirs(os.path.dirname(filename_dest))

            print(filename_dest)
            cv2.imwrite(filename_dest, img1)

def write_csv(csv_file='fovea.csv', source_dir='', fovea_dir=''):
    if os.path.exists(csv_file):
        os.remove(csv_file)

    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')

        csv_writer.writerow(['images', 'masks'])

        for dir_path, subpaths, files in os.walk(source_dir, False):
            for f in files:
                image_file_source = os.path.join(dir_path, f)

                file_base, file_ext = os.path.splitext(image_file_source)  # 分离文件名与扩展名
                if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                    continue

                image_file_fovea = image_file_source.replace(source_dir, fovea_dir)

                csv_writer.writerow([image_file_source, image_file_fovea])

def resize_images_dir(source_dir = '', imgsize=299):
    if not source_dir.endswith('/'):
        source_dir = source_dir + '/'

    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            image_file_source = os.path.join(dir_path, f)

            file_base, file_ext = os.path.splitext(image_file_source)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            img1 = cv2.imread(image_file_source)
            img1 = cv2.resize(img1, (imgsize, imgsize))

            image_file_dest = image_file_source

            print(image_file_dest)

            cv2.imwrite(image_file_dest, img1)

dir_orig = '/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/original'
dir_dest = '/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/fovea'


# op_json('/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/DR0.json',dir_orig, dir_dest, dr_type='DR0')
# op_json('/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/DR1.json',dir_orig, dir_dest, dr_type='DR1')
# op_json('/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/DR2.json',dir_orig, dir_dest, dr_type='DR2')
# op_json('/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/DR3.json',dir_orig, dir_dest, dr_type='DR3')
# op_json('/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/DR4.json',dir_orig, dir_dest, dr_type='DR4')

# write_csv('fovea.csv', dir_orig, dir_dest)

resize_images_dir('/home/ubuntu/Fover_center/DR0_4黄斑不准重新标注400/512', imgsize=512)


print('ok')
