#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 20:16:22 2019

@author: jimmy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 19:58:45 2019

@author: jimmy
"""
import os 
import tensorflow as tf
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
import numpy as np
import re
import pandas as pd


class Params(object):
    def __init__(self):
        # Data Parameters
        self.n = 5000
        self.noise_len = 5
        self.numeric_len = 10
        self.object_len = 4
        self.object_nums = [10,10,10,10]
        
        # Model Parameters
        self.type = 'FM' #LR/FM/FFM
        self.k = 30
        self.epochs = 20
        self.batch_size = 50
        self.optmizer = 'adm' #sgd/adagrad/RMSprop/adam
        self.learning_rate = 0.005
        
        self.l1_reg_rate = 0.005
        self.l2_reg_rate = 0.005
        
        
    def data_dir(self):
        data_dir = f'{os.getcwd()}/data/N{self.n}_{self.noise_len}_{self.numeric_len}_{self.object_len}_{str(self.object_nums)}'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir
    
    def model_path(self):
        model_dir = f'{os.getcwd()}/models/N{self.n}_{self.noise_len}_{self.numeric_len}_{self.object_len}_{str(self.object_nums)}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = f'{self.type}-K{self.k}_E{self.epochs}B{self.batch_size}_O{self.optmizer}_Lr{self.learning_rate}_L1{self.l1_reg_rate}L2{self.l2_reg_rate}.h5'
        return os.path.join(model_dir,model_path)
    
    def fig_dir(self):
        fig_dir = f'{os.getcwd()}/figures/N{self.n}_{self.noise_len}_{self.numeric_len}_{self.object_len}_{str(self.object_nums)}'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_id = f'{self.type}-K{self.k}_E{self.epochs}B{self.batch_size}_O{self.optmizer}_Lr{self.learning_rate}_L1{self.l1_reg_rate}L2{self.l2_reg_rate}'
        return fig_dir,fig_id
    
    def record_dir(self):
        model_dir = f'{os.getcwd()}/records/N{self.n}_{self.noise_len}_{self.numeric_len}_{self.object_len}_{str(self.object_nums)}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir
    
    def show(self):
        format_str = '{:^50}\n'
        print('#'*50+'\n')
        print(format_str.format('Data Parameters'))
        print(f'N:{self.n}, Noise_len:{self.noise_len}, Numeric_len:{self.numeric_len}, Object_len:{self.object_len}, Object_nums:{str(self.object_nums)}\n')
        
        print(format_str.format('Model Parameters'))
        print(f'Model:{self.type}, K:{self.k}, Epochs:{self.epochs}, Batch_size:{self.batch_size}, Optimizer:{self.optmizer}, Lr:{self.learning_rate}, L1:{self.l1_reg_rate}, L2:{self.l2_reg_rate}\n')
        print('#'*50+'\n')

   
def Record(params,performance):
    if not os.path.exists(f'{os.getcwd()}/record'):
        os.makedirs(f'{os.getcwd()}/record')
        
    format_str = '|----{:^25}----|\n'
    #with open(f'{os.getcwd()}/record/record.txt','a+') as f:
    with open(f'{params.record_dir()}/record.txt','a+') as f:
        f.write(format_str.format('Model Location'))
        f.write(f'{params.model_path()}\n')
        
        f.write(format_str.format('Data Description'))
        f.write(f'N:{params.n} Noise_len:{params.noise_len} Numeric_len:{params.numeric_len} Object_len:{params.object_len} Object_nums:{params.object_nums}\n')
        
        f.write(format_str.format('Model Parameters'))
        f.write(f'Model:{params.type}, K:{params.k} Epoch:{params.epochs} Batch_size:{params.batch_size} Optimizer:{params.optmizer} Lr:{params.learning_rate} L1:{params.l1_reg_rate} L2:{params.l2_reg_rate}\n')
        

        f.write(format_str.format('Result Show'))
        
        f.write(performance)
        
    record_to_csv(f'{params.record_dir()}/record.txt',f'{params.record_dir()}/record.csv')
    
  
        
def performance_show(data,model):
    # 训练集预测
    tr_pre_raw,tr_true = model.predict(data.tr_X),data.tr_Y
    f1_dict = {}
    for th in np.arange(0.4,0.6,0.01):
        f1_dict[th] = f1_score(tr_true,np.where(tr_pre_raw>th,1,0))
    th_max = max(f1_dict,key=f1_dict.get)
    tr_pre = np.where(tr_pre_raw>th_max,1,0)
    
    tr_f1 = np.round(f1_score(tr_true,tr_pre),4)
    tr_acc = np.round(accuracy_score(tr_true,tr_pre),4)
    tr_auc = np.round(roc_auc_score(tr_true,tr_pre_raw),4)
    
    # 测试集预测
    te_pre_raw,te_true = model.predict(data.te_X),data.te_Y
    te_pre = np.where(te_pre_raw>th_max,1,0)

    te_f1 = np.round(f1_score(te_true,te_pre),4)
    te_acc = np.round(accuracy_score(te_true,te_pre),4)
    te_auc = np.round(roc_auc_score(te_true,te_pre_raw),4)
    
    row_format = 'In {:<8} F1 score:{:.4f} Acc:{:.4f} AUC:{:.4f} '
    th_format = 'Th_max:{:.2f}'
    row1 = row_format.format('TrainSet',tr_f1,tr_acc,tr_auc)+th_format.format(th_max)
    row2 = row_format.format('TestSet',te_f1,te_acc,te_auc)
    return row1+'\n'+row2+'\n\n\n'


def record_to_csv(record_path,df_name):
    with open(record_path,'r') as f:
        content = f.read()
    
    df_dict = {}
        
    df_dict['model'] = re.findall('Model:(.*?),',content)
    df_dict['k'] = re.findall('K:(.*?) ',content)
    df_dict['epoch'] = re.findall('Epoch:(.*?) ',content)
    df_dict['batch_size'] = re.findall('Batch_size:(.*?) ',content)
    df_dict['optimizer'] = re.findall('Optimizer:(.*?) ',content)
    df_dict['lr'] = re.findall('Lr:(.*?) ',content)
    df_dict['l1'] = re.findall('L1:(.*?) ',content)
    df_dict['l2'] = re.findall('L2:(.*?)\n',content)
    
    f1 = re.findall('F1 score:(.*?) ',content)
    acc = re.findall('Acc:(.*?) ',content)
    auc = re.findall('AUC:(.*?) ',content)
    df_dict['train_f1'] = f1[::2]
    df_dict['train_acc'] = acc[::2]
    df_dict['train_auc'] = auc[::2]
    df_dict['test_f1'] = f1[1::2]
    df_dict['test_acc'] = acc[1::2]
    df_dict['test_auc'] = auc[1::2]
    
    df = pd.DataFrame(df_dict)
    df.to_csv(df_name,index=False)

def print_tf(obj,get=False):
    init_g = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        if get:
            return sess.run(obj)
        print(sess.run(obj))
