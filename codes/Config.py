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
from FFM import FFM
from FM import FM


class Params(object):
    def __init__(self):
        # Model Parameters
        self.k = 30
        self.epoch = 100
        self.learning_rate = 0.0005
        self.batch_size = 50
        self.l1_reg_rate = 0.005
        self.l2_reg_rate = 0.005
        self.optmizer = 'Adam' #GD\Adagrad\Adam
        
        # Data Parameters
        self.n = 20000
        self.noise_len = 10
        self.numeric_len = 10
        self.object_len = 10
        self.object_nums = [4,4,4,4,20,4,4,4,4,20]
        self.data_dir = f'{os.getcwd()}/data/N{self.n}_{self.noise_len}_{self.numeric_len}_{self.object_len}_{str(self.object_nums)}'
    
    def show(self):
        format_str = '{:^50}\n'
        print('#'*50+'\n')
        print(format_str.format('Data Parameters'))
        print(f'N:{self.n}, Noise_len:{self.noise_len}, Numeric_len:{self.numeric_len}, Object_len:{self.object_len}, Object_nums:{str(self.object_nums)}\n')
        
        print(format_str.format('Model Parameters'))
        print(f'K:{self.k}, Optimizer:{self.optmizer}, Epoch:{self.epoch}, Lr:{self.learning_rate}, Batch_size:{self.batch_size}, L1:{self.l1_reg_rate}, L2:{self.l2_reg_rate}\n')
        print('#'*50+'\n')


def get_batch(arr, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < arr.shape[0] else arr.shape[0]
    return arr[start:end]

def print_tf(obj,get=False):
    init_g = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        if get:
            return sess.run(obj)
        print(sess.run(obj))


def R(model,params,data):
    if model == 'FFM':
        model = FFM(params,data)
    elif model == 'FM':
        model = FM(params,data)
    else:
        return 
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model.checkpoint_dir)
        saver.restore(sess,ckpt.model_checkpoint_path)
        

def Restore(model,params,data):
    tf.reset_default_graph()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        if model == 'FFM':
            model = FFM(params,data)
        elif model == 'FM':
            model = FM(params,data)
        else:
            return 

        model.build_model()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
    
        model.restore(sess)
        test_cnt = data.te_X.shape[0]//params.batch_size
        te_P = []
        for j in range(test_cnt):
            batch_X = get_batch(data.te_X, params.batch_size, j)
            result = model.predict(sess, batch_X)[0]
            te_P.extend(result.reshape(-1,).tolist())
        te_pre_raw = np.array(te_P).reshape(-1,)
        te_true = data.te_Y.reshape(-1,)
        
        #compare_df = pd.DataFrame({'real':te_true,'pre':te_pre_raw})
        
        te_pre = np.where(te_pre_raw>th_max,1,0)
        
        f1 = f1_score(te_true,te_pre)
        acc = accuracy_score(te_true,te_pre)
        auc = roc_auc_score(te_true,te_pre_raw)
        print(f'f1 score: {f1},acc: {acc},auc: {auc}')
        Record(params = params,model = model,th_max = th_max,\
               tr_true = tr_true,tr_pre_raw = tr_pre_raw,\
               te_true = te_true,te_pre_raw = te_pre_raw)

    
    
   
def Record(params,model,th_max,tr_true,tr_pre_raw,te_true,te_pre_raw):
    if not os.path.exists(f'{os.getcwd()}/record'):
        os.makedirs(f'{os.getcwd()}/record')
        
    format_str = '|----{:^25}----|\n'
    with open(f'{os.getcwd()}/record/record.txt','a+') as f:
        f.write(format_str.format('Store Location'))
        f.write(f'{model.checkpoint_dir}\n')
        
        f.write(format_str.format('Data Description'))
        f.write(f'N:{params.n} Noise_len:{params.noise_len} Numeric_len:{params.numeric_len} Object_len:{params.object_len} Object_nums:{params.object_nums}\n')
        
        f.write(format_str.format('Model Parameters'))
        f.write(f'K:{params.k} Epoch:{params.epoch} Lr:{params.learning_rate} Optimizer:{params.optmizer} Batch_size:{params.batch_size} L1:{params.l1_reg_rate} L2:{params.l2_reg_rate}\n')
        
        
        tr_pre = np.where(tr_pre_raw>th_max,1,0)
        tr_f1 = np.round(f1_score(tr_true,tr_pre),4)
        tr_acc = np.round(accuracy_score(tr_true,tr_pre),4)
        tr_auc = np.round(roc_auc_score(tr_true,tr_pre_raw),4)
        
        te_pre = np.where(te_pre_raw>th_max,1,0)
        te_f1 = np.round(f1_score(te_true,te_pre),4)
        te_acc = np.round(accuracy_score(te_true,te_pre),4)
        te_auc = np.round(roc_auc_score(te_true,te_pre_raw),4)
        
        f.write(format_str.format('Result Show'))
        f.write(f'In TrainSet F1 score:{tr_f1} Acc:{tr_acc} AUC:{tr_auc} Th_max:{np.round(th_max,2)}\n')
        f.write(f'In TestSet F1 score:{te_f1} Acc:{te_acc} AUC:{te_auc}\n\n\n')