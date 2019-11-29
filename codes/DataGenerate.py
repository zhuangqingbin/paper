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

# def generate_data(n,noise_len,numeric_len,object_len,object_nums,seed,unbalance=False):
#     np.random.seed(seed)
#     x,y = make_classification(n_samples=n,n_features=numeric_len+object_len+noise_len,n_informative=numeric_len+object_len)
#
#     data = pd.DataFrame(x)
#
#     random.seed(seed)
#     object_f_cols = random.sample(range(data.shape[1]),object_len)
#     for _col,_cat in zip(object_f_cols,object_nums):
#         data.iloc[:,_col] = pd.cut(data.iloc[:,_col],_cat,labels=range(_cat)).astype(object)
#
#     data.columns=[f'col_{str(i)}' for i in data.columns]
#     data['label'] = y
#
#     return train_test_split(data,test_size=0.25,random_state=seed)

def generate_cat_data(n, seed, cat_n, cat_format):
    np.random.seed(seed)
    # 均匀分布
    if cat_format == 'uniform':
        return np.random.randint(0, cat_n, size = [n, 1])
    # 泊松分布
    elif cat_format == 'poisson':
        return np.random.poisson(lam = cat_n-1, size = [n, 1])
    # 二项分布
    elif cat_format == 'binomial':
        return np.random.binomial(cat_n-1, 0.5, size = [n, 1])
    else:
        return None

def generate_data(n, seed, means, sigmas, objects_n, objects_format, noise_means, noise_sigmas,
                  format, w_intercept, w_numeric, w_cat,
                  w_intercross_numeric, w_intercross_cat, w_intercross):
    numeric_none = True
    cat_none = True
    noise_none = True
    for mean,sigma in zip(means,sigmas):
        np.random.seed(seed)
        if numeric_none:
            numeric_data = np.random.multivariate_normal(mean = mean, cov = sigma, size = n)
            numeric_none = False
        else:
            numeric_data = np.hstack((numeric_data,
                                      np.random.multivariate_normal(mean = mean, cov = sigma, size = n)))
    numeric_data = pd.DataFrame(numeric_data)

    for mean,sigma in zip(noise_means,noise_sigmas):
        np.random.seed(seed)
        if noise_none:
            noise_data = np.random.multivariate_normal(mean = mean, cov = sigma, size = n)
            noise_none = False
        else:
            noise_data = np.hstack((noise_data,
                                    np.random.multivariate_normal(mean = mean, cov = sigma, size = n)))
    noise_data = pd.DataFrame(noise_data)


    for cat_n,cat_format in zip(objects_n,objects_format):
        if cat_none:
            cat_data = generate_cat_data(n = n,seed = seed,cat_n = cat_n,cat_format = cat_format)
            cat_none = False
        else:
            cat_data = np.hstack((cat_data,
                                 generate_cat_data(n = n,seed = seed,cat_n = cat_n,cat_format = cat_format)))
    cat_data = pd.DataFrame(cat_data, dtype=object)
    one_hot_data = pd.get_dummies(cat_data)

    if format == 'logit':
        phi = 0
        # 数值型变量
        for i, w in enumerate(w_numeric):
            phi += w * numeric_data.iloc[:,i]
        # 分类变量
        np.random.seed(seed)
        w_cat_normal = np.random.normal(w_cat[0], w_cat[1], one_hot_data.shape[1])
        phi += np.sum(one_hot_data.mul(w_cat_normal), axis=1)

        # 交叉项
        ## 数值型交叉项
        w_intercross_numeric_n = np.random.normal(w_intercross_numeric[0], w_intercross_numeric[1],
                                        int(numeric_data.shape[1]*(numeric_data.shape[1]-1)/2))
        index = 0
        for i in range(numeric_data.shape[1]):
            for j in range(i + 1, numeric_data.shape[1]):
                phi += w_intercross_numeric_n[index] * numeric_data.iloc[:, i] * numeric_data.iloc[:, j]
                index += 1
        ## 离散型交叉项
        w_intercross_cat_n = np.random.normal(w_intercross_cat[0], w_intercross_cat[1],
                                        int(one_hot_data.shape[1]*(one_hot_data.shape[1]-1)/2))
        index = 0
        for i in range(one_hot_data.shape[1]):
            for j in range(i + 1, one_hot_data.shape[1]):
                phi += w_intercross_cat_n[index] * one_hot_data.iloc[:, i] * one_hot_data.iloc[:, j]
                index += 1

        ## 数值-离散型交叉项
        w_intercross_n = np.random.normal(w_intercross[0], w_intercross[1],
                                        int(numeric_data.shape[1]*one_hot_data.shape[1]))
        index = 0
        for i in range(numeric_data.shape[1]):
            for j in range(one_hot_data.shape[1]):
                phi += w_intercross_n[index] * numeric_data.iloc[:, i] * one_hot_data.iloc[:, j]
                index += 1
        y = 1/(1+np.exp(-(w_intercept+phi)))
    data = pd.concat((numeric_data,noise_data,cat_data),axis=1)
    data.columns = [f'col_{str(i)}' for i in range(data.shape[1])]
    data['label'] = y.apply(lambda x:np.random.binomial(1,x,1)[0])
    train, test, _, _ = train_test_split(data, data.label, test_size = .2, random_state=seed)
    return train, test


def load_data(params, save = True):
    if not os.listdir(params.data_dir()):
        # train_data,test_data = generate_data(n=params.n,noise_len = params.noise_len,numeric_len=params.numeric_len,object_len=params.object_len,object_nums=params.object_nums,seed=1994)
        train_data, test_data = generate_data(n = params.n, seed = params.seed, means = params.numeric_means,
                           sigmas = params.numeric_sigmas,
                           objects_n = params.objects_n, objects_format = params.objects_format,
                           noise_means = params.noise_means, noise_sigmas = params.noise_sigmas,
                           format = params.model_format, w_intercept = params.w_intercept,
                           w_numeric = params.w_numeric, w_cat = params.w_cat,
                           w_intercross_numeric = params.w_intercross_numeric,
                           w_intercross_cat = params.w_intercross_cat,
                           w_intercross = params.w_intercross)
        if save:
            train_data.to_pickle(f'{params.data_dir()}/train_data.pkl')
            test_data.to_pickle(f'{params.data_dir()}/test_data.pkl')
        else:
            os.rmdir(params.data_dir())
    else:
        print('Load Old Data ......')
        train_data = pd.read_pickle(f'{params.data_dir()}/train_data.pkl')
        test_data = pd.read_pickle(f'{params.data_dir()}/test_data.pkl')
    return train_data,test_data


