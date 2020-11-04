import json
import os
import csv
import cv2
import numpy as np

json_dir = '/media/ubuntu/data2/Fover_center/Floor2/original'
filename_csv = 'macular.csv'

def create_macular(json_dir, filename_csv):

    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])

        for dir_path, subpaths, files in os.walk(json_dir, False):
            for f in files:

                if ('.json' in f) and (not 'result.json' in f) \
                        and (not 'via_region_data (1).json' in f):
                    file_name_json = os.path.join(dir_path, f)

                    # print(file_name_json)

                    with open(file_name_json, 'r') as load_f:
                        dict1 = json.load(load_f)

                        filename_source_image = file_name_json.replace('.json', '')

                        # 个别文件不存在
                        if not os.path.exists(filename_source_image):
                            print('file not exists:', filename_source_image)
                            continue

                        if '/标注软件/' in filename_source_image:
                            continue

                        dirname, filename = os.path.split(filename_source_image)

                        if 'region' in dict1 and len(dict1['region']) > 0:
                            img_original = cv2.imread(filename_source_image)

                            img_macular = np.zeros((img_original.shape[0], img_original.shape[1], 1))

                            for (k, v) in dict1['region'].items():
                                try:
                                    if v['name'] != '黄斑中央凹(Central Foveal Thickness)':
                                        continue
                                except:
                                    continue

                                if len(v['coordinate']) == 0:
                                    continue

                                # region 获取一个个点，生成点的数组
                                array_points = None
                                for i in range(1, len(v['coordinate']) + 1):
                                    scale_ratio = img_original.shape[1] / 1000

                                    x = v['coordinate'][str(i)]['x:'] * scale_ratio
                                    # x= x*10
                                    y = v['coordinate'][str(i)]['y:'] * scale_ratio
                                    # y=y*10
                                    # temp_point = np.array([[int(round(x)), int(round(y))]])
                                    temp_point = np.array([[round(x), round(y)]])

                                    if array_points is None:
                                        array_points = temp_point
                                    else:
                                        array_points = np.concatenate((array_points, temp_point),
                                                                      axis=0)
                                # endregion
                                # points.checkVector(2, CV_32S) >= 0 in function fillConvexPoly
                                # cv2.fillConvexPoly(img_macula, array_points, (255))

                                if array_points is None:
                                    continue

                                cv2.fillConvexPoly(img_macular, np.array(array_points, 'int32'), 255)

                                # 保存病变区域图像文件
                                if np.sum(img_macular > 0) > 0:
                                    filename_name_dest_image = os.path.join(dirname,  'Macular_' + filename)
                                    print(filename_name_dest_image)
                                    csv_writer.writerow([filename_source_image, filename_name_dest_image])
                                    cv2.imwrite(filename_name_dest_image, img_macular)


create_macular(json_dir, filename_csv)

#  remove holes
def remove_hole(base_dir):
    for dir_path, subpaths, files in os.walk(base_dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(f)
            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                continue

            if 'Macula_' in filename:

                print(img_file_source)

                img = cv2.imread(img_file_source)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # cv2.drawContours(img, contours, -1, (255, 255, 255), 1) #区域边缘画线

                for contour in contours:
                    cv2.fillPoly(img, pts=[contour], color=(255, 255, 255))

                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(img_file_source, gray_image)


# remove_hole(json_dir)



print('OK')
