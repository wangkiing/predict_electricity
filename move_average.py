# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:11:33 2018

@author: wangkiing
"""

import numpy as np;
import pandas as pd;
import matplotlib.pylab as plt;
from sklearn.model_selection import train_test_split;
from dateutil.relativedelta import relativedelta
from sklearn import metrics;


class move_average(object):
    
    def __init__(self,data,ma_size):
        self.data = data;
        self.model = None;
        self.train_x = None;
        self.test_x = None;
        self.weight ={};
        self.predict = {};
        self.ma_size =ma_size;
        self.epsilon  = 0.0001;
        #梯度下降迭代轮数
        self.max_itor = 1000;
        
     #即将数据分为训练数据和测试数据
    def get_train_and_test_data(self):
        #75%作为测试数据
        train_index = int(len(self.data)*0.75)+1;
        self.train_x = self.data[:train_index];
        #25%的数据作为测试数据
        test_index = int(len(self.data)*0.25)+train_index+1;
        self.test_x = self.data[train_index:test_index];
    
    #加权移动平均预测
    def weight_move_average_fit(self,end,learn_rate,weight):
        for ma in self.ma_size:
            select_index = 0;
#            self.gradient_boosting(self.train_x,learn_rate,ma)
            while True:
                select_x = self.train_x[self.train_x.index[select_index:select_index+ma]];
                select_x_matrix=select_x.as_matrix();
                weight_x = weight*select_x_matrix;
                series = pd.Series(np.sum(weight_x),index=[select_x.index[-1]]);
                series = self.reindex_predict_data(series);
                if not ('WMA'+str(ma) in self.predict):
                    self.predict['WMA'+str(ma)] = series;
                else:
                    self.predict['WMA'+str(ma)]=self.predict['WMA'+str(ma)].append(series);
                if(self.predict['WMA'+str(ma)].index[-1]==end):
                    break;
                select_index = select_index+1;
#            print('------------------------------')
#            print(self.predict['WMA'+str(ma)])
    
    #通过随机梯度下降寻找最优解
#    def gradient_boosting(self,data,learn_rate,ma):
#        weights  = np.zeros(ma+1);
#        error0=0;
#        count=0
#        while True:
#            count=count+1;
#            select_index = 0;
#            for m in range(len(self.train_x)-12):
#                #获取参数
#                select_x = self.train_x[self.train_x.index[select_index:select_index+ma]];
#                select_x_matrix=select_x.as_matrix();
#                #增加x0=1项
#                select_x_matrix= np.append(select_x_matrix,[1],axis=0) 
#                weight_x = weights*select_x_matrix;
#                #计算残差
#                diff = np.sum(weight_x)/ma - self.train_x[self.train_x.index[select_index+ma]];
#                # 梯度 = diff[0] * x[i][j]  
#                weights -=learn_rate*diff*select_x_matrix;
#                pd.ewma
#                select_index = select_index+1;
#                
#            
#            #计算MSE
#            select_index = 0;
#            for i in range(len(self.train_x)-1):
#                #获取参数
#                select_x = self.train_x[self.train_x.index[select_index:select_index+ma]];
#                select_x_matrix=select_x.as_matrix();
#                #增加x0=1项
#                select_x_matrix= np.append(select_x_matrix,[1],axis=0)
#                weight_x = weights*select_x_matrix;
#                #获取y的index
#                series = pd.Series(np.sum(weight_x)/ma,index=[select_x.index[-1]]);
#                series = self.reindex_predict_data(series);
#                #计算残差
#                error1 = (np.sum(weight_x)/ma - self.train_x[series.index[-1]])**2/ma;
#            if error1 -error0<self.epsilon or count>=self.max_itor:
#                break;
#            else:
#                error0 = error1;
#        print("MA %d weight is: "%(ma));
#        print(weights)
        
    #移动平均预测
    def move_average(self,end):
        self.get_train_and_test_data();
        tarin_x = self.train_x;
        for ma in self.ma_size:
            while True:
                self.predict['MA'+str(ma)] = pd.rolling_mean(tarin_x,ma).dropna();
                self.predict['MA'+str(ma)] = self.reindex_predict_data(self.predict['MA'+str(ma)]);
                if(self.predict['MA'+str(ma)].index[-1] == end):
                    break;
                tarin_x[self.predict['MA'+str(ma)].index[-1]] = self.predict['MA'+str(ma)][-1];
    
    #获取预测和原数据图像
    def get_picture(self):
        for predict_index in self.predict:
            plt.plot(self.predict[predict_index],label='predict-'+predict_index,color=[np.random.rand(),np.random.rand(),np.random.rand()]);
        plt.plot(self.train_x,label='origin',color='red');
        plt.legend();
        plt.show();
    
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
    
    #重新建立索引
    def reindex_predict_data(self,data):
        rein = [];
        for predict_index in data.index:
            rein.append(predict_index+relativedelta(months=1));
        return pd.Series(data.tolist(),index=rein);
        
if __name__ == '__main__':
     #日期转换
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m');
    #读取数据
    cvs_file = 'F:\\启明星\\人工智能\\51101_data.csv';
    #读入数据
    csv_series = pd.read_csv(cvs_file,encoding='gbk',index_col='年月',parse_dates=['年月'],date_parser=dateparse);
    csv_series.dropna(inplace=True)
    x = csv_series['energy\r售电量'];
    model = move_average(x,[4]);
    model.move_average(end = pd.Timestamp(2017,7,1));
    model.weight_move_average_fit(end = pd.Timestamp(2017,7,1),learn_rate=0.1,weight=[0.1,0.2,0.2,0.5]);
    model.get_picture();
    model.get_RMS();
