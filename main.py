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
from params import Params,performance_show,Record,record_to_csv,print_tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

########################################################
############## Data Load
########################################################
params = Params()
train_data,test_data = load_data(params)
data = Data(train_data,test_data,'label')


def main(params, data):
    params.show()
    model = common_model(params,data)
    model.save(params.model_path())
    #print(model.summary())
    performance = performance_show(data,model)
    print('\n'+performance)
    
    Record(params,performance)
    return model
#model = main(params,data)




def run():
    for t in ['LR','FM','FFM','FFM-None']:
        #for o in ['sgd','adagrad','RMSprop','adam']:
        for o in ['adam']:
            for lr in [0.005]:
                params.type = t
                params.optmizer = o
                params.learning_rate = lr
                model = main(params,data)
run()             
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