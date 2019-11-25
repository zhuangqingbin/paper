#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:18:31 2019

@author: jimmy
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import random,os

def generate_data(n,noise_len,numeric_len,object_len,object_nums,seed,unbalance=False):
    np.random.seed(seed)
    x,y = make_classification(n_samples=n,n_features=numeric_len+object_len+noise_len,n_informative=numeric_len+object_len)
    
    data = pd.DataFrame(x)
    
    random.seed(seed)
    object_f_cols = random.sample(range(data.shape[1]),object_len)
    for _col,_cat in zip(object_f_cols,object_nums):
        data.iloc[:,_col] = pd.cut(data.iloc[:,_col],_cat,labels=range(_cat)).astype(object)
        
    data.columns=[f'col_{str(i)}' for i in data.columns]
    data['label'] = y
    
    return train_test_split(data,test_size=0.25,random_state=seed)


def load_data(params):
    if not os.listdir(params.data_dir()):
        train_data,test_data = generate_data(n=params.n,noise_len = params.noise_len,numeric_len=params.numeric_len,object_len=params.object_len,object_nums=params.object_nums,seed=1994)
        train_data.to_pickle(f'{params.data_dir()}/train_data.pkl')
        test_data.to_pickle(f'{params.data_dir()}/test_data.pkl')
    else:
        train_data = pd.read_pickle(f'{params.data_dir()}/train_data.pkl')
        test_data = pd.read_pickle(f'{params.data_dir()}/test_data.pkl')
    return train_data,test_data
    
    
