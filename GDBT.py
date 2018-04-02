# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:53:41 2018

@author: wangkiing
"""

import numpy as np;
from sklearn import metrics  
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd;
import matplotlib.pylab as plt;
from sklearn.model_selection import train_test_split;
from sklearn.datasets import load_boston;
from sklearn.metrics import mean_squared_error;


"""初始化算法，设置参数

一些主要参数
loss: 损失函数，GBDT回归器可选'ls', 'lad', 'huber', 'quantile'。
learning_rate: 学习率/步长。
n_estimators: 迭代次数，和learning_rate存在trade-off关系。
criterion: 衡量分裂质量的公式，一般默认即可。
subsample: 样本采样比例。
max_features: 最大特征数或比例。

决策树相关参数包括max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, 
max_leaf_nodes, min_impurity_split, 多数用来设定决策树分裂停止条件。
"""
def get_model(train_x,train_y):
    model = GradientBoostingRegressor(loss='ls', learning_rate=0.02, n_estimators=200, subsample=1,max_depth=4);
    model.fit(train_x,train_y);
    return model;

#获取训练及测试数据
def read_data():
    #读取数据
    cvs_file = 'C:\\Users\\wangkiing\Desktop\\人工智能\\51101_data.csv';
#    cvs_file = 'C:\\Users\\wangkiing\Desktop\\人工智能\\pollution.csv';
    #读入数据
    csv_series = pd.read_csv(cvs_file,encoding='gbk');
    csv_series.dropna(inplace=True)
    x = csv_series.drop('energy\r售电量',axis=1);
#    x = csv_series.drop(['pollution','date'],axis=1);
    x= x.as_matrix();
    y = csv_series['energy\r售电量'];
#    y = csv_series['pollution'];
    y= y.as_matrix();
    #50%的数据作为训练数据
    train_index = int(len(x)*0.9)+1;
    train_x = x[:train_index];
    train_y = y[:train_index];
    #25%的数据作为测试数据
    test_index = int(len(x)*0.1)+train_index+1;
    test_x = x[train_index:test_index];
    test_y = y[train_index:test_index];
    #25%的数据作为验证数据
    valid_x = x[test_index:];
    vaild_y = y[test_index:];
    return train_x,train_y,test_x,test_y,valid_x,vaild_y;
if __name__ =='__main__':
    #读入数据并处理
    train_x,train_y,test_x,test_y,valid_x,vaild_y=read_data();
#    train_x,test_x,train_y,test_y=train_test_split(load_boston().data, load_boston().target, test_size=0.2);
    #训练模型
    model = get_model(train_x,train_y);
    #测试训练集
    predict_train=model.predict(train_x);
    rmse_train = mean_squared_error(train_y, predict_train);
    #测试测试集
    predict_test=model.predict(test_x);
    rmse_test = mean_squared_error(test_y,predict_test);
    
    #画出预测图像和测试数据图像
    print ("RMSE for training dataset is %f, for testing dataset is %f." % (np.sqrt(rmse_train), np.sqrt(rmse_test)))
