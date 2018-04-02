# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:44:32 2018

@author: wangkiing
"""

import pandas as pd;
from sklearn.linear_model import LinearRegression;
from sklearn.model_selection import train_test_split;
from sklearn.model_selection import cross_val_predict
from sklearn import metrics;
import numpy as np;
import matplotlib.pylab as plt;

class liner_regression(object):
    
    def __init__(self,x,y):
        self.x = x;
        self.y = y;
        self.train_x =None;
        self.train_y =None;
        self.test_x = None;
        self.test_y = None;
        self.fit_model = None;
        
    #训练模型
    def fit(self):
        self.get_train_and_test_data(self.x,self.y);
        model = LinearRegression();
        model.fit(self.train_x,self.train_y);
        self.fit_model = model;
    
    #查看均方根差，判断模型好坏
    def get_RSM(self):
        train_predict = self.fit_model.predict(self.train_x);
        test_predict = self.fit_model.predict(self.test_x);
        train_MSE = metrics.mean_squared_error(train_predict,self.train_y);
        test_MSE = metrics.mean_squared_error(test_predict,self.test_y);
        print("train RSM:",np.sqrt(train_MSE));
        print("test RSM:",np.sqrt(test_MSE))
    
    #即将数据分为训练数据和测试数据
    def get_train_and_test_data(self,serise_x,serise_y):
        train_x,test_x,train_y,test_y = train_test_split(serise_x,serise_y);
        self.train_x = train_x;
        self.train_y=train_y;
        self.test_x = test_x;
        self.test_y = test_y;
    
    #获取匹配的模型
    def get_model(self):
        return self.fit_model;
    
    #交叉验证优化
    def cross_validation(self):
        self.get_train_and_test_data(self.x,self.y);
        self.predict = cross_val_predict(self.fit_model,self.x,self.y);
        MSE = metrics.mean_squared_error(self.predict,self.y);
        print("RSM:",np.sqrt(MSE));
    
    def get_predit_data_picture(self):
        fig, ax = plt.subplots()
        ax.scatter(y, self.predict)
        ax.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'k--', lw=4)
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        plt.show()
        
if __name__=='__main__':
    #数据导入及处理
#    file_path = 'D:\\pyworkspace\\predict_electricity\\Folds5x2_pp.csv';
#    csv_data = pd.read_csv(file_path);
#    x = csv_data[['AT', 'V', 'AP', 'RH']];
#    y = csv_data['PE'];
    
    cvs_file = 'C:\\Users\\wangkiing\Desktop\\人工智能\\51101_data.csv';
    csv_series = pd.read_csv(cvs_file,encoding='gbk');
    #去除NA值
    csv_series.dropna(inplace=True)
    x = csv_series.drop('energy\r售电量',axis=1);
    #dataform转矩阵
    x= x.as_matrix();
    y = csv_series['energy\r售电量'];
    y= y.as_matrix();

    model = liner_regression(x,y);
    model.fit();
    model.get_RSM();
    
    #采用交叉验证模型的RSE比未采用的大，主要原因是我们这里是对所有折的样本做测试集对应的预测值的RSE，
    #而之前仅仅对25%的测试集做了MSE。两者的先决条件并不同。
    model.cross_validation();
    model.get_predit_data_picture();