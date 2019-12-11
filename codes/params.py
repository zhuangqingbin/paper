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

param_dict ={
    'target':'simulate',
    'n': 20000,
    'seed': 1994,
    # data parameters
    'numeric_n': 20,
    'm':4,
    'alpha1':5,
    'cat_n':4,
    'alpha2':1,
    'alpha3':1,
    'alpha4':1,
    'alpha5':1,
    'noise_n':5,
    'alpha6':1,
    'model_format':'logit',
    # model parameters
    'type': 'FM',
    'k': 20,
    'epochs': 100,
    'batch_size': 100,
    'optmizer': 'adam',
    'learning_rate': 0.001,
    'l1_reg_rate': 0.001,
    'l2_reg_rate': 0.001
}



class Params(object):
    def __init__(self, param_dict):
        self.target = param_dict['target']

        # Data Parameters
        self.n = param_dict['n']
        self.seed = param_dict['seed']

        self.numeric_n = param_dict['numeric_n']

        self.m = param_dict['m']
        self.alpha1 = param_dict['alpha1']

        self.cat_n = param_dict['cat_n']
        self.alpha2 = param_dict['alpha2']

        self.alpha3 = param_dict['alpha3']
        self.alpha4 = param_dict['alpha4']

        self.alpha5 = param_dict['alpha5']
        self.noise_n = param_dict['noise_n']
        self.alpha6 = param_dict['alpha6']
        self.model_format = param_dict['model_format']
        
        # Model Parameters
        self.type = param_dict['type']
        self.k = param_dict['k']
        self.epochs = param_dict['epochs']
        self.batch_size = param_dict['batch_size']
        self.optmizer = param_dict['optmizer']   #sgd/adagrad/RMSprop/adam
        self.learning_rate = param_dict['learning_rate']
        
        self.l1_reg_rate = param_dict['l1_reg_rate']
        self.l2_reg_rate = param_dict['l2_reg_rate']

    def data_id(self):
        # Data Identifier
        if self.target == 'simulate':
            return f'N{self.n}_S{self.seed}_Cn{self.numeric_n}_m{self.m}_a1-{self.alpha1}_' \
                f'On{self.cat_n}_a2-{self.alpha2}_a3-{self.alpha3}_a4-{self.alpha4}_' \
                f'a5-{self.alpha5}_Nn{self.noise_n}_a6-{self.alpha6}'
        else:
            return self.target

    def data_dir(self):
        data_dir = f'{os.getcwd()}/data/{self.data_id()}'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir
    
    def model_path(self):
        model_dir = f'{os.getcwd()}/models/{self.data_id()}'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = f'{self.type}-K{self.k}_E{self.epochs}B{self.batch_size}_O{self.optmizer}_Lr{self.learning_rate}_L1{self.l1_reg_rate}L2{self.l2_reg_rate}.h5'
        return os.path.join(model_dir,model_path)

    
    def fig_dir(self):
        fig_dir = f'{os.getcwd()}/figures/{self.data_id()}'

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_id = f'{self.type}-K{self.k}_E{self.epochs}B{self.batch_size}_O{self.optmizer}_Lr{self.learning_rate}_L1{self.l1_reg_rate}L2{self.l2_reg_rate}'
        return fig_dir,fig_id
    
    def record_dir(self):
        model_dir = f'{os.getcwd()}/records/{self.data_id()}'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def data_info(self):
        info = f'N:{self.n}, model_format:{self.model_format}, Seed:{self.seed} \n' + \
        f'Numeric nums:{self.numeric_n}, ' + \
        f'Numeric groups:{self.m}, ' + \
        f'Numeric coef:{self.alpha1} \n' + \
        f'Object nums:{self.cat_n}, ' + \
        f'Object coef:{self.alpha2} \n' + \
        f'Numeric intercept:{self.alpha3}, ' + \
        f'Object intercept:{self.alpha4}, ' + \
        f'Both intercept:{self.alpha5} \n' + \
        f'Noise nums:{self.noise_n}, ' + \
        f'Noise coef:{self.alpha6} \n'
        return info


    def show(self):
        """
        :return: 按照type打印数据和模型的信息
        """
        format_str = '{:^50}\n'
        print('#'*50+'\n')
        if self.target == 'simulate':
            print(format_str.format('Data Parameters'))
            print(self.data_info())
        else:
            print(format_str.format('Data Info'))
            print('Demonstration using real data.')

        print(format_str.format('Model Parameters'))
        print(f'Model:{self.type}, K:{self.k}, Epochs:{self.epochs}, Batch_size:{self.batch_size} \n'
              f'Optimizer:{self.optmizer}, Lr:{self.learning_rate}, L1:{self.l1_reg_rate}, L2:{self.l2_reg_rate}\n')
        print('#'*50+'\n')

   
def Record(params,performance):
    if not os.path.exists(f'{os.getcwd()}/records'):
        os.makedirs(f'{os.getcwd()}/records')
        
    format_str = '|----{:^25}----|\n'
    #with open(f'{os.getcwd()}/record/record.txt','a+') as f:
    with open(f'{params.record_dir()}/record.txt','a+') as f:
        f.write(format_str.format('Model Location'))
        for i,item in enumerate(re.sub('(.*?)/paper/', '', params.model_path()).split('/'),1):
            f.write(f'Folder level{i}:{item}\n')

        f.write(format_str.format('Data Description'))
        if params.target == 'simulate':
            f.write(params.data_info())
        else:
            f.write('Demonstration using real data. \n')

        f.write(format_str.format('Model Parameters'))
        f.write(f'Model:{params.type}, K:{params.k} Epoch:{params.epochs} Batch_size:{params.batch_size} \n' +
                f'Optimizer:{params.optmizer} Lr:{params.learning_rate} L1:{params.l1_reg_rate} L2:{params.l2_reg_rate} ')
        

        f.write(format_str.format('Result Show'))
        
        f.write(performance)
        
    record_to_csv(params)
    
  
        
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



def record_to_csv(params):

    with open(f'{params.record_dir()}/record.txt','r') as f:
        content = f.read()
    
    df_dict = {}
        
    df_dict['model'] = re.findall('Model:(.*?),',content)
    df_dict['k'] = re.findall('K:(.*?) ',content)
    df_dict['epoch'] = re.findall('Epoch:(.*?) ',content)
    df_dict['batch_size'] = re.findall('Batch\_size:(.*?) ',content)
    df_dict['optimizer'] = re.findall('Optimizer:(.*?) ',content)
    df_dict['lr'] = re.findall('Lr:(.*?) ',content)
    df_dict['l1'] = re.findall('L1:(.*?) ',content)
    df_dict['l2'] = re.findall('L2:(.*?) ',content)
    
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


    if params.target == 'simulate':
        data_path = f'{params.data_dir()}'
    else:
        data_path = f'{os.getcwd()}/empirical/' + params.target
    train_data = pd.read_pickle(os.path.join(data_path,'train_data.pkl'))
    test_data = pd.read_pickle(os.path.join(data_path, 'test_data.pkl'))
    tr_1per = (train_data.label == 1).mean().round(4)
    te_1per = (test_data.label == 1).mean().round(4)
    df['tr_1per'] = tr_1per
    df['te_1per'] = te_1per

    df.to_csv(f'{params.record_dir()}/record.csv',index=False)

def print_tf(obj,get=False):
    init_g = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        if get:
            return sess.run(obj)
        print(sess.run(obj))
