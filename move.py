import zipfile
import shutil
import os

zip_file = './dataset/traindata.zip'
extract_folder = './dataset'
target_folder = './dataset'

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# 移动文件夹到目标位置
shutil.move(os.path.join(extract_folder, 'ir'), os.path.join(target_folder, 'ir'))
shutil.move(os.path.join(extract_folder, 'vi'), os.path.join(target_folder, 'vi'))
