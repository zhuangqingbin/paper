#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 16:03:33 2019

@author: jimmy
"""
import sys,os
sys.path.append('%s/codes' % os.getcwd())
sys.path.append('%s/data' % os.getcwd())
from DataProcess import Data
from DataGenerate import load_data
from Models import common_model,get_model
from params import param_dict,Params
from params import performance_show,Record,record_to_csv,print_tf
import os
import pandas as pd
import numpy as np
import itertools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

########################################################
############## Data Load
########################################################
def main(params, data):
    params.show()
    model = common_model(params, data)
    model.save(params.model_path())
    performance = performance_show(data, model)
    print('\n' + performance)

    Record(params, performance)
    return model

def run(param_dict):
    params = Params(param_dict)
    if param_dict['target'] == 'simulate':
        train_data, test_data = load_data(params)
        data = Data(train_data, test_data, 'label')
    else:
        train_data = pd.read_pickle('train_data.pkl')
        test_data = pd.read_pickle('test_data.pkl')
        data = Data(train_data, test_data, 'label')
    main(params,data)

for seed in range(1990,2020):
    for optmizer in ['sgd','adagrad','RMSprop','adam']:
        for k in [30,40]:
            for type in ['LR','LR-R','FM','FM-R']:
                param_dict['seed'] = seed
                param_dict['optmizer'] = optmizer
                param_dict['k'] = k
                param_dict['type'] = type
                run(param_dict=param_dict)




def get_params(target = 'simulate'):
    params = Params(target = target)
    if target == 'simulate':
        train_data, test_data = load_data(params)
        data = Data(train_data, test_data, 'label')
    else:
        train_data = pd.read_pickle('train_data.pkl')
        test_data = pd.read_pickle('test_data.pkl')
        data = Data(train_data, test_data, 'label')
    return params,data

def run_old(target='simulate'):
    params, data = get_params(target)
    for lr in [0.005, 0.001]:
        #for o in ['sgd','adagrad','RMSprop','adam']:
        for o in ['adam']:
            for t in ['LR', 'LR-R', 'FM', 'FM-R']:

                params, data = get_params(target)
                params.type = t
                params.optmizer = o
                params.learning_rate = lr
                model = main(params,data)


params_dict1 = {
    'k': [20,25,30,35,40,45,50],
    'epochs': [100,150,200],
    'optmizer': ['sgd','adagrad','RMSprop','adam'],
    'learning_rate': [0.001,0.005,0.01,0.05,0.1],
    'l1_reg_rate': [0.001,0.005,0.01],
    'l2_reg_rate': [0.001,0.005,0.01],
    'type': ['LR','LR-R','FM','FM-R']
}

def train(target, params_dict):
    params, data = get_params(target)
    for item in list(itertools.product(*params_dict.values())):
        params.k = item[0]
        params.epochs = item[1]
        params.optmizer = item[2]
        params.learning_rate = item[3]
        params.l1_reg_rate = item[4]
        params.l2_reg_rate = item[5]
        params.type = item[6]
        model = main(params, data)
#train(target='shizheng',params_dict=params_dict)
#run()
#model = main(params,data)

#if __name__ == '__main__':
#    x_train, x_test, y_train, y_test = data.tr_X,data.te_X,data.tr_Y,data.te_Y
#    
#    
#    lr_model = common_model(x_train,x_test,y_train,y_test,'LR',train=True)
#    print(lr_model.summary())
#    
#    fm_model = common_model(x_train,x_test,y_train,y_test,'FM',train=True)
#    print(fm_model.summary())
#    
#    
#    ffm_model = common_model(x_train,x_test,y_train,y_test,'FFM',train=True,data=data)
#    print(ffm_model.summary())
#    
#    
#    LRNN_tr_f1 = f1_score(data.tr_Y,np.where(lr_model.predict(data.tr_X)>0.5,1,0))
#    LRNN_te_f1 = f1_score(data.te_Y,np.where(lr_model.predict(data.te_X)>0.5,1,0))
#    
#    FM_tr_f1 = f1_score(data.tr_Y,np.where(fm_model.predict(data.tr_X)>0.5,1,0))
#    FM_te_f1 = f1_score(data.te_Y,np.where(fm_model.predict(data.te_X)>0.5,1,0))
#    
#    FFM_tr_f1 = f1_score(data.tr_Y,np.where(ffm_model.predict(data.tr_X)>0.5,1,0))
#    FFM_te_f1 = f1_score(data.te_Y,np.where(ffm_model.predict(data.te_X)>0.5,1,0))
#    
#    
#    LR = LogisticRegression()
#    LR.fit(data.tr_X,data.tr_Y)
#    LR_tr_f1 = f1_score(data.tr_Y,LR.predict(data.tr_X))
#    LR_te_f1 = f1_score(data.te_Y,LR.predict(data.te_X))
#    
#    format_str = '{:<4} Model: f1 in TrainSet:{:.4f},f1 in TestSet:{:.4f}'
#    
#    print(format_str.format('LR',LR_tr_f1,LR_te_f1))
#    print(format_str.format('LRNN',LRNN_tr_f1,LRNN_tr_f1))
#    print(format_str.format('FM',FM_tr_f1,FM_te_f1))
#    print(format_str.format('FFM',FFM_tr_f1,FFM_te_f1))