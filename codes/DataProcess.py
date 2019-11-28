#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:01:28 2019

@author: jimmy
"""
import numpy as np
import pandas as pd

class Data(object):
    """
    输入含有label的训练集和测试集，
    提取数据集的f、feature2field，
    将分类变量独热，
    tr_X、tr_Y为训练集数据和标签，te_X、te_Y为测试集数据和标签（数组格式）
    """
    def __init__(self,train_data,test_data,label):
        self.tr_Y = train_data[label].values.reshape(-1,1)
        self.te_Y = test_data[label].values.reshape(-1,1)
        
        data = pd.concat([train_data,test_data],axis=0)
        
        features = data.drop([label],axis=1).columns.tolist()
        self.f = len(features)

        data_list = []
        self.fields_dict,self.feature2field = {},{}
        cur_index = 0
        for f_index,f_name in enumerate(features):
            if data[f_name].dtype == object:
                unique_values = data[f_name].unique()
                len_unique = len(unique_values)
                tmp_dict = dict(zip(unique_values,range(cur_index,cur_index+len_unique)))
                
                for i in range(cur_index,cur_index+len_unique):
                    self.feature2field[i] = f_index
                tmp_data = pd.get_dummies(data[f_name])
                tmp_data.rename(tmp_dict,axis=1,inplace=True)
                data_list.append(tmp_data)
                cur_index += len_unique
                
            else:
                self.feature2field[cur_index] = f_index
                tmp_data = data[[f_name]]
                tmp_data.rename({f_name:cur_index},axis=1,inplace=True)
                data_list.append(tmp_data)
                cur_index += 1
        self.p = len(self.feature2field)
        data = pd.concat(data_list,axis=1)
        self.tr_X = data.iloc[:train_data.shape[0],:].values
        self.te_X = data.iloc[train_data.shape[0]:,:].values
