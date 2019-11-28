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
print(train_data.info())
print(train_data.label.value_counts(normalize=True))

print(train_data.col_10.nunique())
data = Data(train_data,test_data,'label')

print(data.tr_X.shape)
print(data.tr_Y.shape)

from sklearn import linear_model, datasets
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(data.tr_X, data.tr_Y)

prepro = logreg.predict_proba(data.te_X)
acc = logreg.score(data.te_X,data.te_Y)
print('\n\nLR acc :', acc)
