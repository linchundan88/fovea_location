import os
import csv
import pandas as pd
import sklearn

def write_csv(filename_csv, list_image_file, list_x, list_y):
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'x', 'y'])

        for (image_file, x, y) in zip(list_image_file, list_x, list_y):
            csv_writer.writerow([image_file, x, y])


def split_dataset(filename_csv, valid_ratio=0.1, test_ratio=None,
                  shuffle=True, random_state=None, field_columns=['images', 'x', 'y']):
    
    df = pd.read_csv(filename_csv)
    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    if test_ratio is None:
        split_num = int(len(df) * (1 - valid_ratio))
        data_train = df[:split_num]
        list_train_image = data_train[field_columns[0]].tolist()
        list_train_x = data_train[field_columns[1]].tolist()
        list_train_y = data_train[field_columns[2]].tolist()

        data_valid = df[split_num:]
        list_valid_image = data_valid[field_columns[0]].tolist()
        list_valid_x = data_valid[field_columns[1]].tolist()
        list_valid_y = data_valid[field_columns[2]].tolist()

        return list_train_image, list_train_x, list_train_y, list_valid_image, list_valid_x, list_valid_y
    else:
        split_num_train = int(len(df) * (1 - valid_ratio - test_ratio))
        data_train = df[:split_num_train]
        list_train_image = data_train[field_columns[0]].tolist()
        list_train_x = data_train[field_columns[1]].tolist()
        list_train_y = data_train[field_columns[2]].tolist()

        split_num_valid = int(len(df) * (1 - test_ratio))
        data_valid = df[split_num_train:split_num_valid]
        list_valid_image = data_valid[field_columns[0]].tolist()
        list_valid_x = data_valid[field_columns[1]].tolist()
        list_valid_y = data_valid[field_columns[2]].tolist()

        data_test = df[split_num_valid:]
        list_test_image = data_test[field_columns[0]].tolist()
        list_test_x = data_test[field_columns[1]].tolist()
        list_test_y = data_test[field_columns[2]].tolist()

        return list_train_image, list_train_x, list_train_y, \
               list_valid_image, list_valid_x, list_valid_y,\
               list_test_image, list_test_x, list_test_y



