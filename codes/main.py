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
import pickle
import sys,os

sys.path.append('%s/codes' % os.getcwd())
sys.path.append('%s/data' % os.getcwd())

from Config import Params,get_batch
from DataProcess import Data
from DataGenerate import generate_data
from FFM import FFM
from Logistic import Logistic

from sklearn.metrics import f1_score,accuracy_score,roc_auc_score


train_data_path = '%s/data/train_sample.csv' % os.getcwd()
#train_data = pd.read_csv(train_data_path,dtype={'C1':object,'C18':object,'C16':object})
#test_data = pd.read_csv(train_data_path,dtype={'C1':object,'C18':object,'C16':object})


train_data,test_data = generate_data(n=20000,numeric_len=10,object_len=5,object_nums=[4,4,4,4,4],seed=1994)

# loading base params
params = Params()
data = Data(train_data,test_data,'label')    



gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Session(config=gpu_config) as sess:
    model = FFM(params,data)
    model.build_model()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    if params.is_training:
        cnt = data.tr_X.shape[0]//params.batch_size
        for e in range(params.epoch):
            for j in range(cnt):
                batch_X = get_batch(data.tr_X,params.batch_size,j)
                batch_y = get_batch(data.tr_Y,params.batch_size,j)
                loss = model.train(sess, batch_X, batch_y)
                if (j+1) % 10 == 0:
                    print(f'Epoch {e+1} batch {j+1} loss {loss}')
        model.save(sess, params.checkpoint_dir)

    #else:
        model.restore(sess, params.checkpoint_dir)
        cnt = data.te_X.shape[0]//params.batch_size
        te_P = []
        for j in range(cnt):
            batch_X = get_batch(data.te_X, params.batch_size, j)
            result = model.predict(sess, batch_X)[0]
            te_P.extend(result.reshape(-1,).tolist())
        te_P = np.array(te_P).reshape(-1,)
        compare_df1 = pd.DataFrame({'real':data.te_Y.reshape(-1,),'pre':te_P})
        y_true = data.te_Y.reshape(-1,)
        y_pre = np.where(te_P>0.3,1,0)
        
        f1 = f1_score(y_true,y_pre)
        acc = accuracy_score(y_true,y_pre)
        auc = roc_auc_score(y_true,te_P)
        print(f'f1 score: {f1},acc: {acc},auc: {auc}')
 



