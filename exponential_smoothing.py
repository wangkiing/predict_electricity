# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:19:19 2018

@author: wangkiing
"""
import numpy as np;
from sklearn import metrics;
import pandas as pd;
from dateutil.relativedelta import relativedelta
import matplotlib.pylab as plt;


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
        
    def first_fit(self):
        t=0;
        self.get_train_and_test_data();
        while t< len(self.train_x):
            if t==0:
                ma = self.first_es_data(self.train_x,None,t)
                self.predict['first_es'] = pd.Series(ma,index = [self.train_x.index[0]]);
            else:
                t_index = self.train_x.index[t];
                ma = self.first_es_data(self.train_x,self.predict['first_es'],t)
                self.predict['first_es'][t_index] = ma;
            t+=1;
    #一次指数平滑
    def first_es_data(self,data,predict_data,t):
        if t==0:
             ma = np.sum(data)/len(data);
             return ma;
        else:
            t_1_index = data.index[t-1];
            ma = self.alpha*data[t_1_index]+(1-self.alpha)*predict_data[t_1_index];
            return ma;
    
    #二次指数平滑
    def second_es_data(self,data,predict_data,t):
        #一次平滑
        if not 'first_es' in self.predict.keys():
            self.first_fit();
        #二次平滑
        if t==0:
             ma = np.sum(data)/len(data);
             return ma;
        else:
            t_1_index = data.index[t-1];
            ma = self.alpha*data[t_1_index]+(1-self.alpha)*predict_data[t_1_index];
            return ma;
        
    def second_fit(self):
        t=0;
        self.get_train_and_test_data();
        while t< len(self.train_x):
            if t==0:
                ma = self.first_es_data(self.train_x,None,t)
                self.predict['second_es'] = pd.Series(ma,index = [self.train_x.index[0]]);
            else:
                t_index = self.predict['first_es'].index[t];
                ma = self.second_es_data(self.predict['first_es'],self.predict['second_es'],t)
                self.predict['second_es'][t_index] = ma;
            t+=1;
        
     #三次指数平滑
    def third_es_data(self,data,predict_data,t):
        #一次平滑
        if not 'second_es' in self.predict.keys():
            self.second_fit();
        #二次平滑
        if t==0:
             ma = np.sum(data)/len(data);
             return ma;
        else:
            t_1_index = data.index[t-1];
            ma = self.alpha*data[t_1_index]+(1-self.alpha)*predict_data[t_1_index];
            return ma;
        
    def third_fit(self):
        t=0;
        self.get_train_and_test_data();
        while t< len(self.train_x):
            if t==0:
                ma = self.first_es_data(self.train_x,None,t)
                self.predict['third_es'] = pd.Series(ma,index = [self.train_x.index[0]]);
            else:
                t_index = self.predict['second_es'].index[t];
                ma = self.second_es_data(self.predict['second_es'],self.predict['third_es'],t)
                self.predict['third_es'][t_index] = ma;
            t+=1;
            
    
    #预测
    def predic_data(self,end):
        for predict_index in self.predict:
            while True:
                if(self.predict[predict_index].index[-1] == end):
                    break;
                predict_data = self.predict[predict_index];
                if predict_index == 'first_es':
                    tempdata =self.first_es_data(self.train_x,self.predict[predict_index],0);
                elif predict_index == 'second_es':
                    tempdata =self.second_es_data(self.predict['first_es'],self.predict[predict_index],0);
                elif predict_index == 'third_es':
                    tempdata =self.third_es_data(self.predict['second_es'],self.predict[predict_index],0);
                self.predict[predict_index] = self.add_new_data(predict_data,tempdata);

    #获取训练集和测试集的RMS
    def get_RMS(self):
        for predict_index in self.predict:
            temp_data = self.predict[predict_index];
            train_predict_x = temp_data[self.train_x.index].dropna();
            test_predict_x = temp_data[self.test_x.index].dropna();
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

     #获取预测和原数据图像
    def get_picture(self):
        for predict_index in self.predict:
            plt.plot(self.predict[predict_index],label='predict-'+predict_index,color=[np.random.rand(),np.random.rand(),np.random.rand()]);
        plt.plot(self.series,label='origin',color='red');
        plt.legend();
        plt.show();
    
if __name__ =='__main__':
     #日期转换
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m');
    #读取数据
    cvs_file = 'F:\\启明星\\人工智能\\51101_data.csv';
    #读入数据
    csv_series = pd.read_csv(cvs_file,encoding='gbk',index_col='年月',parse_dates=['年月'],date_parser=dateparse);
    csv_series.dropna(inplace=True)
    x = csv_series['energy\r售电量'];
    
    model = es_model(x,0.6);
    model.first_fit();
    model.second_fit();
    model.third_fit();
    model.predic_data(pd.Timestamp(2015,7,1));
    model.get_RMS();
    model.get_picture();