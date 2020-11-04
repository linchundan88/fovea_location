import os
import csv
import cv2

dir_preprocess = '/media/ubuntu/data2/Fover_center/Floor2/preprocess512/'
dir_tmp = '/tmp2/Floor2/'

preprocess_image_size = 512

filename_csv = os.path.join(os.path.abspath('.'), 'Floor2.csv')
if os.path.exists(filename_csv):
    os.remove(filename_csv)

with open(filename_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['images', 'x', 'y'])

    for dir_path, subpaths, files in os.walk(dir_preprocess, False):
        for f in files:
            full_filename = os.path.join(dir_path, f)

            dir_base, filename = os.path.split(full_filename)
            file_basename, file_ext = os.path.splitext(filename)
            if file_ext.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            if file_basename.startswith('Macular'):
                continue

            file_macular = os.path.join(dir_base, 'Macular_'+filename)
            if not os.path.exists(file_macular):
                continue

            img1 = cv2.imread(file_macular)
            img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 1:
                continue

            M = cv2.moments(contours[0])
            x = M["m10"] / M["m00"]
            y = M["m01"] / M["m00"]
            # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            csv_writer.writerow([full_filename, round(x, 2), round(y, 2)])
            print(file_macular)

            if 'dir_tmp' in dir():
                img_preprocess = cv2.imread(full_filename)
                (height, width) = img_preprocess.shape[0:2]
                thickness = int(max(img_preprocess.shape[:-1]) / 100)
                img_draw_circle = cv2.circle(img_preprocess, (int(x), int(y)), thickness, (0, 0, 255), -1)

                image_file_tmp = full_filename.replace(dir_preprocess, dir_tmp)
                os.makedirs(os.path.dirname(image_file_tmp), exist_ok=True)
                cv2.imwrite(image_file_tmp, img_draw_circle)
                print(image_file_tmp)

            # cv2.findContours

            '''
            circles = cv2.HoughCircles(img1, method=cv2.HOUGH_GRADIENT, dp=1,
                minDist=50, param1=150, param2=5, minRadius=5, maxRadius=40)
            if circles is None:
                print(file_macular)
            if len(circles[0]) == 1:
                x, y, r = circles[0][0]
                print(x, y)
            else:
                print(file_macular)
            '''