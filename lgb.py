#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 16:43:50 2019

@author: jimmy
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import f1_score,mean_squared_error
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

class lgb_class(object):
    '''
    Examples
    --------
    params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'num_boost_round':1000,
    'learning_rate':0.001,
    
    'num_leaves':32,
    'max_depth':7,
    'min_data_in_leaf': 200,
    
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1994,
    'bagging_fraction': 0.9,
    'bagging_freq': 10,
    'bagging_seed': 1994,
    'early_stopping_round': 50,
    'lambda_l1': 0,
    'lambda_l2': 0,
    
    'metric': ['binary_logloss','auc'],
    'is_unbalance': True,
    'verbose': 1}

    gbm = lgb_class(params=params,N=5,random_state=2019)
    gbm.fit(train=train_data,test=test_data,label='target',data_id='id',
    threshold_list=np.arange(40,60)/100)
    '''
    def __init__(self,params,N,random_state):
        self.params = params
        self.N = N
        self.random_state = random_state
        self.train_loss = []
        self.importance = []
        self.test_submit = []
        self.dict_scores = {}
        self.imp_df = pd.DataFrame()
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    def train(self,X_tr,Y_tr,X_va,Y_va):
        print('Train beginning.....')
        lgb_train = lgb.Dataset(X_tr.values,Y_tr.values,feature_name=self.feature_names, categorical_feature=self.object_names)
        lgb_eval = lgb.Dataset(X_va.values,Y_va.values,feature_name=self.feature_names, categorical_feature=self.object_names,reference=lgb_train)
        
        gbm = lgb.train(self.params,
                        lgb_train,
                        num_boost_round=5000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=50,
                        verbose_eval=200,
                        feature_name=self.feature_names,
                        categorical_feature=self.object_names
                        )
        return gbm

    def split(self,N,random_state):
        skf = KFold(n_splits = N,random_state = random_state, shuffle=True)
        #skf = StratifiedKFold(n_splits=N, random_state=random_state, shuffle=True)
        return skf
    
    def criterion(self,y_true, y_pred):
        return f1_score(y_true, y_pred)
    
    
    def show_best(self,y_true):
        for t in self.threshold_list:
            pre = np.where(self.fit_result>t,1,0)
            score = self.criterion(y_true,pre)
            self.dict_scores[str(t)] = score
            print('Thershold {} f1-score:{}'.format(t,score))
    
        self.max_t = float(max(self.dict_scores,key=self.dict_scores.get))
        self.max_score = self.dict_scores[str(self.max_t)]
    
    def show_loss(self):
        self.loss = np.mean(self.train_loss)
        print('Train_data binary_logloss:', self.loss)

    
    def show_imp_features(self,data,top=5):
        im = 0
        for i in self.importance:
            im = im+i
        im = im/self.N
        self.imp_df['feature'] = list(data.columns)
        self.imp_df['importance'] = im
        self.imp_df.sort_values('importance',ascending=False,inplace=True)
        self.imp_features = self.imp_df['feature'].iloc[:top]
        print(f'Top{top} important features:'+str(list(self.imp_features)))
        
        
    def predict(self,data_id):
        predict_result = 0
        for pre in self.test_submit:
            predict_result += pre
        
        self.predict_result = predict_result/self.N
        te_pre = pd.Series(list(self.predict_result)).apply(lambda x: 1 if x >self.max_t else 0)
    
        self.result_df = pd.DataFrame({'id':data_id,'pre':te_pre})
        
        
    def write(self,base_path = 'submit'):
        ### test_data预测并写出csv
        id = datetime.datetime.strftime(datetime.datetime.now(),'%m%d%H%M')
        file = '%s%s' % (base_path,datetime.datetime.strftime(datetime.datetime.now(),'%m%d'))
        if os.path.exists(file):
            pass
        else:
            os.mkdir(file)
        
        path_text = '%s/record.txt' % file
        with open(path_text,'a+') as f:
            f.write('[Classfier]ID:{} Train Score:{} Per:{} N:{} p:{}\n '.format(id,round(self.max_score,4),round((self.result_df.pre==1).mean(),4),self.N,len(self.imp_df)))
            f.write('{} of {} best in train_data\n '.format(self.max_t,str(self.threshold_list))) 
            f.write('Params:'+str(self.params)+'\n')
            f.write('Im_features:'+str(list(self.imp_features))+'\n\n')
        
        path_csv = '{}/Classifier{}.csv'.format(file,id) 
        self.result_df.to_csv(path_csv,index = False)
    
    def play_imp(self,save=False):
        plt.figure(figsize=(8, 10))
        sns.barplot(x="importance", y="feature",data=self.imp_df)
        plt.title(f'LightGBM Features (avg over {self.N}folds)')
        plt.tight_layout()
        if save:
            plt.savefig('lgbm_importances.png')
    
    def fit(self,train,test,label,data_id,threshold_list):
        print('Train beginning.....')
        # 训练集数据
        X_train = train.drop([label,data_id], axis = 1)
        # 训练集标签
        Y_train = train[label]
        # 测试集数据
        drop_cols = []
        if label in test.columns:
            drop_cols.append(label)
        if data_id in test.columns:
            drop_cols.append(data_id)
        X_test = test.drop(drop_cols, axis = 1)
        
        # 区分类别变量
        self.feature_names = X_train.columns.tolist()
        self.object_names = []
        for _f in self.feature_names:
            if X_train[_f].dtype == object:
                self.object_names.append(_f)


        self.fit_result = np.zeros(len(X_train))
        
        self.threshold_list = threshold_list
        
        print('='*50)
        print('The shape of train_data:',X_train.shape)
        print('The shape of train_label:',Y_train.shape)
        print('The shape of test_data:',X_test.shape)
        print('='*50)
        
        skf = self.split(N=self.N,random_state=self.random_state)
        for k, (tr_in, va_in) in enumerate(skf.split(X_train, Y_train)):
            print('-'*50)
            print('Train {} flod begins.....'.format(k+1))
            # 按照CV划分出训练集数据/标签、测试集数据/标签
            X_tr, X_va, Y_tr, Y_va = X_train.iloc[tr_in,:], X_train.iloc[va_in,:], Y_train.iloc[tr_in], Y_train.iloc[va_in]
            # 利用划分的数据和标签训练出模型
                    
            gbm = self.train(X_tr,Y_tr,X_va,Y_va)
            
            # 利用模型预测验证集的结果
            va_p = gbm.predict(X_va.values,num_iteration=gbm.best_iteration)
            self.fit_result[va_in] = va_p
            # 利用模型预测测试集的结果
            pre_p = gbm.predict(X_test.values, num_iteration=gbm.best_iteration)
            # 训练的metric误差
            metric = self.params['metric']
            loss = gbm.best_score['valid_0'][metric] if type(metric)==str\
                         else gbm.best_score['valid_0'][metric[0]]
                        
            self.train_loss.append(loss)
            self.importance.append(gbm.feature_importance())
            self.test_submit.append(pre_p)
        
        # 展示threshold_list中的t和score
        print('-'*50+'\n')
        print('Traverse threshold_list.\n')
        self.show_best(train[label].values)
        
        # 展示loss
        print('-'*50+'\n')
        self.show_loss()
        
        # 展示重要性变量
        print('-'*50+'\n')
        print('Important Features.\n')
        self.show_imp_features(data=X_train)
        
        # 预测
        self.predict(test[data_id].values)
        
        # 结果输出
        self.write()
train_data['data_id'] = range(len(train_data))
test_data['data_id'] = range(len(test_data))


params = {
'objective': 'binary',
'boosting': 'gbdt',
'num_boost_round':1000,
'learning_rate':0.1,

'num_leaves':32,
'max_depth':7,
'min_data_in_leaf': 200,

'feature_fraction': 0.9,
'feature_fraction_seed': 1994,
'bagging_fraction': 0.9,
'bagging_freq': 10,
'bagging_seed': 1994,
'early_stopping_round': 50,
'lambda_l1': 0,
'lambda_l2': 0,

'metric': ['binary_logloss','auc'],
'is_unbalance': True,
'verbose': 1}
gbm = lgb_class(params=params,N=5,random_state=2019)
gbm.fit(train=train_data,test=test_data,label='label',data_id='data_id',threshold_list=np.arange(40,60)/100)