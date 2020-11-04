import os
from LIBS.Data.my_data import split_dataset, write_csv

data_version = 'V3'
data_type = 'fovea_adults_ROP'
filename_csv = os.path.join(os.path.abspath('../'), 'datafiles', data_version, f'{data_type}.csv')
filename_csv_train = os.path.join(os.path.abspath('../'), 'datafiles', data_version, f'{data_type}_train.csv')
filename_csv_valid = os.path.join(os.path.abspath('../'), 'datafiles', data_version, f'{data_type}_valid.csv')
filename_csv_test = os.path.join(os.path.abspath('../'), 'datafiles', data_version, f'{data_type}_test.csv')

list_train_image, list_train_x, list_train_y, \
list_valid_image, list_valid_x, list_valid_y, \
list_test_image, list_test_x, list_test_y = \
    split_dataset(filename_csv, valid_ratio=0.1, test_ratio=0.1, field_columns=['images', 'x', 'y'])

write_csv(filename_csv_train, list_train_image, list_train_x, list_train_y)
write_csv(filename_csv_valid, list_valid_image, list_valid_x, list_valid_y)
write_csv(filename_csv_test, list_test_image, list_test_x, list_test_y)

print('OK')

