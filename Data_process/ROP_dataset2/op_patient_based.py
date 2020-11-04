import os
import shutil

dir_source = '/media/ubuntu/data2/Fover_center/ROP/2020_11_4/original'
dir_dest = '/media/ubuntu/data2/Fover_center/ROP/2020_11_4_patient_based/original'
list_patients = []


#'00a7fce6cd000fe54d65811fd86f7e74c3c086bc#00a7fce6cd000fe54d65811fd86f7e74c3c086bc#206e45eb-c145-4f28-bfd7-2fda32bb7a79.11.json
for dir_path, subpaths, files in os.walk(dir_source, False):
    for f in files:
        full_filename = os.path.join(dir_path, f)
        file_base, file_ext = os.path.splitext(full_filename)
        if file_ext.lower() not in ['.json', '.jpg', '.jpeg', 'png']:
            continue

        if file_ext.lower() in ['.json']:
            s_check_id = file_base.split('#')[-1]
            print(s_check_id)
            list1 = s_check_id.split('.')
            patient_id = ''.join(list1[:-1])
            if patient_id not in list_patients:
                list_patients.append(patient_id)

                file_dest = full_filename.replace(dir_source, dir_dest)
                print(file_dest)
                os.makedirs(os.path.dirname(file_dest), exist_ok=True)
                shutil.copy(full_filename, file_dest)

                shutil.copy(full_filename.replace('.json', '.jpg'),
                            file_dest.replace('.json', '.jpg'))
            else:
                print(f'{patient_id}patient already exist.')

print('OK')