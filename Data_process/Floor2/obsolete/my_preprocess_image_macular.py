import cv2
import numpy as np
import os
from LIBS.ImgPreprocess import my_preprocess


def do_process_dir(source_dir, dest_dir, list_imagesize=[299, 384, 512]):
    print('preprocess start')
    # 去掉最后一个字符
    if source_dir.endswith('/'):
        source_dir = source_dir[:-1]
    if dest_dir.endswith('/'):
        dest_dir = dest_dir[:-1]


    # for i_image_size in [299]:
    for i_image_size in list_imagesize:
        # 由于有多级目录
        dest_dir = os.path.join(dest_dir, str(i_image_size))

        for dir_path, subpaths, files in os.walk(source_dir, False):

            for f in files:
                img_file_source = os.path.join(dir_path, f)

                filename, file_extension = os.path.splitext(f)
                if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                    # print('file ext name:', f)
                    continue

                if 'Macular_' in filename:
                    continue

                # 图像文件对应的黄斑文件存在
                dirname, filename = os.path.split(img_file_source)

                filename_macular = os.path.join(dirname, 'Macular_' + filename)
                if not os.path.exists(filename_macular):
                    continue

                print(img_file_source)

                #region 处理原始图像
                (found_circle, x, y, r) = my_preprocess.detect_xyr(img_file_source)
                img1 = my_preprocess.my_crop_xyr(img_file_source, x, y, r)

                # 增加一些黑边， imgaug进行crop,rotate防止删除有意义区域
                img1 = my_preprocess.add_black_margin(img1, add_black_pixel_ratio=0.06)

                img_file_dest = img_file_source.replace(source_dir, dest_dir)
                if not os.path.exists(os.path.dirname(img_file_dest)):
                    os.makedirs(os.path.dirname(img_file_dest))

                img1 = cv2.resize(img1, (i_image_size, i_image_size))
                cv2.imwrite(img_file_dest, img1)
                #endregion

                #region 处理Macular文件
                img1 = my_preprocess.my_crop_xyr(filename_macular, x, y, r)

                img1 = my_preprocess.add_black_margin(img1, add_black_pixel_ratio=0.06)

                img1 = img1.astype(np.uint8)
                gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                ret, img1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                img_file_dest1 = filename_macular.replace(source_dir, dest_dir)
                if not os.path.exists(os.path.dirname(img_file_dest1)):
                    os.makedirs(os.path.dirname(img_file_dest1))

                img1 = cv2.resize(img1, (i_image_size, i_image_size))
                cv2.imwrite(img_file_dest1, img1)
                #endregion

    print('preprocess end')


source_dir = '/home/jsiec/disk1/PACS/DR-粗标/preprocess/'

dest_dir = '/home/jsiec/disk1/PACS/DR-粗标/preprocess_macular_384'
do_process_dir(source_dir, dest_dir, list_imagesize=[384])

dest_dir = '/home/jsiec/disk1/PACS/DR-粗标/preprocess_macular_512'
do_process_dir(source_dir, dest_dir, list_imagesize=[512])



print('OK')
