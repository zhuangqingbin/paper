import os
import shutil

dirs = ['data', 'figures', 'models', 'records']

def delete_dir(path):
    print('Delete %s...' % path)
    shutil.rmtree(path)  # 能删除该文件夹和文件夹下所有文件
    os.mkdir(path)

for dir in dirs:
    delete_dir(os.path.join(os.getcwd(),dir))
