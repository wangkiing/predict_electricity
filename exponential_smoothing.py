# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:19:19 2018

@author: wangkiing
"""
import numpy as np;
from sklearn import metrics;
import pandas as pd;
from dateutil.relativedelta import relativedelta

class es_model(object):
    
    def __init__(self,serise,alpha):
        self.series = serise;
        self.alpha=alpha;
        self.predict = {};
        self.train_x =None;
        self.test_x = None;
        self.model = None
     #即将数据分为训练数据和测试数据
    def get_train_and_test_data(self):
        #75%作为测试数据
        train_index = int(len(self.series)*0.75)+1;
        self.train_x = self.series[:train_index];
        #25%的数据作为测试数据
        test_index = int(len(self.series)*0.25)+train_index+1;
        self.test_x = self.series[train_index:test_index];
        
    def fit(self):
        t=0;
        self.get_train_and_test_data();
        while t< len(self.train_x):
            if t==0:
                ma = np.sum(self.train_x)/len(self.train_x);
                self.predict['first_es'] = pd.Series(ma,index = [self.train_x.index[0]]);
            else:
                t_1_index = self.train_x.index[t-1];
                t_index = self.train_x.index[t];
                series = pd.Series(self.alpha*self.train_x[t_1_index]+(1-self.alpha)*self.predict['first_es'][t_1_index],index = [t_index]);
                self.predict['first_es']=series;
            t+=1;

    def predict(self,end):
        while True:
            if(self.predict.index[-1] == end):
                break;
            for predict_index in self.predict:
                predict_data = self.predict[predict_index];
                t_1_index = self.predict.index[t-1];
                tempdata = self.alpha*self.train_x[t_1_index]+(1-self.alpha)*self.predict[predict_index][t_1_index];
                self.predict[predict_index] = self.add_new_data(predict_data,tempdata);

    #获取训练集和测试集的RMS
    def get_RMS(self):
        for predict_index in self.predict:
            temp_data = self.predict[predict_index];
            train_predict_x = temp_data[self.train_x.index].dropna();
            test_predict_x = temp_data[self.test_x.index[:-1]].dropna();
            train_MSE = metrics.mean_squared_error(train_predict_x,self.train_x[train_predict_x.index]);
            test_MSE = metrics.mean_squared_error(test_predict_x,self.test_x[test_predict_x.index]);
            print("------------move average--------------");
            print("train %s RSM:%f" %(predict_index,np.sqrt(train_MSE)));
            print("test %s RSM:%f" %(predict_index,np.sqrt(test_MSE)));
            
     #添加新数据
    def add_new_data(self,orgin_data,new_data,type='month'):
        if type=='day':
            new_index = orgin_data.index[-1]+relativedelta(days=1);
        elif type=='month':
            new_index = orgin_data.index[-1]+relativedelta(months=1);
        orgin_data[new_index] = new_data;
        return orgin_data;


if __name__ =='__main__':
     #日期转换
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m');
    #读取数据
    cvs_file = 'F:\\启明星\\人工智能\\51101_data.csv';
    #读入数据
    csv_series = pd.read_csv(cvs_file,encoding='gbk',index_col='年月',parse_dates=['年月'],date_parser=dateparse);
    csv_series.dropna(inplace=True)
    x = csv_series['energy\r售电量'];
    
    model = es_model(x,0.8);
    model.fit();
    model.get_RMS();