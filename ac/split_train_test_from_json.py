
import shutil
import json
import os
from pathlib import Path


def split_from_partition_json(json_path,data_path):
    #open json file
    file = open(json_path)
    dic = json.load(file)
    ext = '.jpg'

    new_folder_name = json_path.split('/')[-1][:-5]
    new_folder_parent = str(Path(json_path).parent.absolute())+'/'+new_folder_name

    if not os.path.isdir(new_folder_parent):
        os.mkdir(new_folder_parent)

    for key in dic.keys():
        file_ls = dic[key]
        #make dir
        folder_key = new_folder_parent+'/'+key

        if not os.path.isdir(folder_key):
            os.mkdir(folder_key)
        
        for file in file_ls:
            src=data_path+'/'+file+ext
            dest=folder_key+'/'+file+ext
            shutil.copyfile(src,dest)
            
    return

if __name__ == '__main__':
    json_dir =  "/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/partitions/"
    file_name = "partition_alpha_angle_0.7_0.15_0.15_0.06552.json" 
    json_path = json_dir+file_name

    data_path = "/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/images/img"

    split_from_partition_json(json_path,data_path)