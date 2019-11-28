#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 16:03:33 2019

@author: jimmy
"""
import sys,os
import warnings
warnings.filterwarnings('ignore')

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
def summary(data):
    count_dict = dict()
    for i in range(data.shape[1]):
        if data.iloc[:,i].dtype == int:
            count_dict['int'] = count_dict.get('int',0) + 1
        if data.iloc[:,i].dtype == float:
            count_dict['float'] = count_dict.get('float',0) + 1
        if data.iloc[:,i].dtype == object:
            count_dict['object'] = count_dict.get('object',0) + 1
    count_dict['shape'] = data.shape
    return count_dict



params = Params()
train_data,test_data = load_data(params,save = False)

print('Train Dataset Shape:{shape}, float:{float}, object:{object}, int:{int}.'.format(**summary(train_data)))
print('Test Dataset Shape:{shape}, float:{float}, object:{object}, int:{int}.'.format(**summary(test_data)))
print('Train Dataset Distribution:')
print(train_data.label.value_counts(normalize=True))
print('Test Dataset Distribution:')
print(test_data.label.value_counts(normalize=True))

data = Data(train_data,test_data,'label')
print('One-hot Train Dataset Shape:')
print(data.tr_X.shape)
print('One-hot Train Dataset Shape:')
print(data.te_X.shape)

from sklearn import linear_model, datasets
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(data.tr_X, data.tr_Y)

prepro = logreg.predict_proba(data.te_X)
acc = logreg.score(data.te_X,data.te_Y)
print('\n\nLR acc :', acc)
