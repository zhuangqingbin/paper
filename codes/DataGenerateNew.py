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
        return np.random.poisson(lam = cat_n, size = [n, 1])
    # 二项分布
    elif cat_format == 'binomial':
        return np.random.binomial(cat_n-1, 0.5, size = [n, 1])
    else:
        return None


def generate_data(n, seed, numeric_n, m, alpha1, cat_n, alpha2, alpha3,
                  alpha4, alpha5, noise_n,alpha6):
    numeric_none = True
    cat_none1 = True
    cat_none2 =True


    # 生成数值型变量，numeric_n个数值型变量，分为m组
    k = int(numeric_n/m)
    # 按照规则生成多元正态分布的协方差矩阵
    sigma = np.eye(k,k)
    for i,v in enumerate(sigma):
        for j,_ in enumerate(v):
            sigma[i][j] = 0.5**(abs(i-j))

    for i in range(m):
        np.random.seed(i*3+100)
        if numeric_none:
            numeric_data = np.random.multivariate_normal(mean = np.zeros(k), cov = sigma, size = n)
            numeric_none = False
        else:
            tmp_data = np.random.multivariate_normal(mean = np.zeros(k), cov = sigma, size = n)
            numeric_data = np.hstack((numeric_data,tmp_data))
    numeric_data = pd.DataFrame(numeric_data)

    # 生成多分类变量，cat_n个多分类变量，分为2组
    t = int(cat_n/2)
    # 1组为均匀整数分类变量
    for i in range(t):
        if cat_none1:
            cat_data1 = generate_cat_data(n = n, seed = i*3+100,
                                          cat_n = 20+5*i,
                                          cat_format='uniform')
            cat_none1 = False
        else:
            tmp_data = generate_cat_data(n = n, seed = i*3+100,
                                          cat_n = 20+5*i,
                                          cat_format = 'uniform')
            cat_data1 = np.hstack((cat_data1,tmp_data))
    # 1组为泊松分布分类变量
    for i in range(t):
        if cat_none2:
            cat_data2 = generate_cat_data(n = n, seed = i*3+100,
                                          cat_n = 10+5*i,
                                          cat_format='poisson')
            cat_none2 = False
        else:
            tmp_data = generate_cat_data(n = n, seed = i*3+100,
                                         cat_n = 10+5*i,
                                         cat_format = 'poisson')
            cat_data2 = np.hstack((cat_data2,tmp_data))
    # 多分类变量合并在一起
    cat_data = pd.DataFrame(np.hstack((cat_data1,cat_data2)), dtype=object)
    # 独热编码
    one_hot_data = pd.get_dummies(cat_data)

    # 生成噪声变量，独立分布，服从均值为0，方差为alpha6的正态分布
    noise_data = np.random.multivariate_normal(mean = np.zeros(noise_n),
                                  cov = np.eye(noise_n,noise_n)*alpha6,
                                  size = n)
    noise_data = pd.DataFrame(noise_data)

    # 按照逻辑回归函数生成y
    #if format == 'logit':
    if True:
        phi = 0
        # 数值型变量
        np.random.seed(seed)
        w_numeric_normal = np.random.randint(-alpha1,alpha1,numeric_data.shape[1])
        phi += np.sum(numeric_data.mul(w_numeric_normal), axis=1)
        # 分类变量
        np.random.seed(seed)
        w_cat_normal = np.random.normal(0, alpha2, one_hot_data.shape[1])
        phi += np.sum(one_hot_data.mul(w_cat_normal), axis=1)

        # 交叉项
        ## 数值型交叉项
        np.random.seed(seed)
        w_intercross_numeric_n = np.random.normal(0, alpha3,
                                        int(numeric_data.shape[1]*(numeric_data.shape[1]-1)/2))
        #w_intercross_numeric_n = np.where(abs(w_intercross_numeric_n) < w_intercross_numeric[2],
        #                                  0, w_intercross_numeric_n)
        index = 0
        for i in range(numeric_data.shape[1]):
            for j in range(i + 1, numeric_data.shape[1]):
                phi += w_intercross_numeric_n[index] * numeric_data.iloc[:, i] * numeric_data.iloc[:, j]
                index += 1
        ## 离散型交叉项
        np.random.seed(seed)
        w_intercross_cat_n = np.random.normal(0, alpha4,
                                        int(one_hot_data.shape[1]*(one_hot_data.shape[1]-1)/2))
        #w_intercross_cat_n = np.where(abs(w_intercross_cat_n) < w_intercross_cat[2],
        #                                  0, w_intercross_cat_n)
        index = 0
        for i in range(one_hot_data.shape[1]):
            for j in range(i + 1, one_hot_data.shape[1]):
                phi += w_intercross_cat_n[index] * one_hot_data.iloc[:, i] * one_hot_data.iloc[:, j]
                index += 1

        ## 数值-离散型交叉项
        np.random.seed(seed)
        w_intercross_n = np.random.normal(0, alpha5,
                                        int(numeric_data.shape[1]*one_hot_data.shape[1]))
        #w_intercross_n = np.where(abs(w_intercross_n) < w_intercross[2],
        #                              0, w_intercross_n)
        index = 0
        for i in range(numeric_data.shape[1]):
            for j in range(one_hot_data.shape[1]):
                phi += w_intercross_n[index] * numeric_data.iloc[:, i] * one_hot_data.iloc[:, j]
                index += 1
        y = 1/(1+np.exp(-(1+phi)))

    data = pd.concat((numeric_data,noise_data,cat_data),axis=1)
    data.columns = [f'col_{str(i)}' for i in range(data.shape[1])]
    data['label'] = y.apply(lambda x:np.random.binomial(1,x,1)[0])
    train, test, _, _ = train_test_split(data, data.label, test_size = .2, random_state=seed)
    return train, test


def load_data(params, save = True):
    if not os.listdir(params.data_dir()):
        # train_data,test_data = generate_data(n=params.n,noise_len = params.noise_len,numeric_len=params.numeric_len,object_len=params.object_len,object_nums=params.object_nums,seed=1994)
        train_data, test_data = \
            generate_data(n = params.n, seed = params.seed,
                          numeric_n = params.numeric_n, m = params.m,
                          alpha1 = params.alpha1, cat_n = params.cat_n,
                          alpha2 = params.alpha2, alpha3 = params.alpha3,
                          alpha4 = params.alpha4, alpha5 = params.alpha5,
                          noise_n = params.noise_n,alpha6 = params.alpha6)
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
