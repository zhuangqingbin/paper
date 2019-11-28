import os
import shutil

dirs = ['data', 'figures', 'models', 'records']

def delete_dir(path):
    shutil.rmtree(dirpath)  # 能删除该文件夹和文件夹下所有文件
    os.mkdir(dirpath)

for dir in dirs:
    delete_dir(os.path.join(os.getcwd(),dir))
