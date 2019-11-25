#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 19:10:12 2019

@author: jimmy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:32:05 2019

@author: jimmy
"""

import tensorflow as tf
tf.reset_default_graph()
import numpy as np
import pandas as pd
import sys,os

sys.path.append('%s/codes' % os.getcwd())
sys.path.append('%s/data' % os.getcwd())

from Config import Params,get_batch,Record,Restore
from DataProcess import Data
from DataGenerate import load_data
from FFM1 import FFM1
from FM import FM
from FFM import FFM
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score

########################################################
############## Data Load
########################################################
params = Params()
train_data,test_data = load_data(params)
data = Data(train_data,test_data,'label') 
   




########################################################
############## Model Build
########################################################
def main(params=params,data=data):
    params.show()
    
    tf.reset_default_graph()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        model = FFM(params,data)
        model.build_model()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
    
        train_cnt = data.tr_X.shape[0]//params.batch_size
        flag = 0
        for e in range(params.epoch):
            if flag:
                break
            for j in range(train_cnt):
                batch_X = get_batch(data.tr_X,params.batch_size,j)
                batch_y = get_batch(data.tr_Y,params.batch_size,j)
                loss = model.train(sess, batch_X, batch_y)
                if np.isnan(loss):
                    flag = 1
                    break
                if (j+1) % 10 == 0:
                    print('Epoch {:<3d} Batch {:<3d} Loss {:.4f}'.format(e+1,j+1,loss))
            
                    
        model.save(sess)
        
        # fit
        tr_P = []
        for t in range(train_cnt):
            batch_X = get_batch(data.tr_X, params.batch_size, t)
            result = model.predict(sess, batch_X)[0]
            tr_P.extend(result.reshape(-1,).tolist())
        tr_pre_raw = np.array(tr_P).reshape(-1,)
        tr_true = data.tr_Y.reshape(-1,)
        
        f1_dict = {}
        for th in np.arange(0.4,0.6,0.01):
            f1_dict[th] = f1_score(tr_true,np.where(tr_pre_raw>th,1,0))
        th_max = max(f1_dict,key=f1_dict.get)
        
        # predict
        # model.restore(sess)
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
        
        f1 = np.round(f1_score(te_true,te_pre),4)
        acc = np.round(accuracy_score(te_true,te_pre),4)
        auc = np.round(roc_auc_score(te_true,te_pre_raw),4)
        print(f'f1 score: {f1},acc: {acc},auc: {auc}')
        Record(params = params,model = model,th_max = th_max,\
               tr_true = tr_true,tr_pre_raw = tr_pre_raw,\
               te_true = te_true,te_pre_raw = te_pre_raw)

for o in ['GD','Adagrad','Adadelta']:
    #params.k = K
    params.epoch =50
    params.optmizer = o
    #params.learning_rate = lr
    main(params=params,data=data)
#
#
#
## Restore(model='FFM',params=params,data=data)
#
#
#
#from sklearn.linear_model import LogisticRegression
#LR = LogisticRegression()
#LR.fit(data.tr_X,data.tr_Y)
#tr_pre = LR.predict(data.tr_X)
#te_pre = LR.predict(data.te_X)
#tr_f1 = f1_score(data.tr_Y,tr_pre)
#te_f1 = f1_score(data.te_Y,te_pre)
