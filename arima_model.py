# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:52:37 2018

@author: wangkiing
"""

import pandas as pd;
import matplotlib.pylab as plt;
import numpy as np;
from statsmodels.tsa.stattools import adfuller;
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA;
from statsmodels.stats.diagnostic import acorr_ljungbox;
import warnings;
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.relativedelta import relativedelta


class ARIMA_Model(object):
    
    def __init__(self,time_series,maxlag=9):
        self.time_series = time_series;
        self.maxlag = maxlag;
        self.p=0;
        self.q=0;
        self.RMSE_data=[];
        self.proper_model = None;
        self.predict_ts = None;
        self.bic = {'bic':1000000,'p':0,'q':0};
        self.aic = {'aic':1000000,'p':0,'q':0};
        
    #计算最优ARIAM模型
    def get_proper_model(self):
       with warnings.catch_warnings():
           warnings.filterwarnings("ignore")
           for p in range(self.maxlag):
               for q in range(self.maxlag):
                   model = ARMA(self.time_series,order=(p,q));
                   try:
                       model_result = model.fit(disp=-1,method='css');
                   except:
                       continue;
                   if model_result.bic <self.bic['bic']:
                       self.bic = {'bic':model_result.bic,'p':p,'q':q};
                       self.q =q;
                       self.p =p;
                       self.proper_model = model_result;
                       self.predict_ts = model_result.predict();
                       
                   if  model_result.aic <self.aic['aic']:
                       self.aic = {'aic':model_result.aic,'p':p,'q':q};
                   
    #自相关和偏自相关图
    def draw_acf_pacf(self):
        f =plt.figure(facecolor='white');
        ax1 = f.add_subplot(211);
        plot_acf(self.time_series,ax=ax1);
        ax2 = f.add_subplot(212);
        plot_pacf(time_series,ax=ax2);
        plt.legend(loc='best')
        plt.show();
        
    #,残差分析 正态分布 QQ图线性诊断
    def draw_resid_qq(self):
        self.proper_model.plot_diagnostics(figsize=(15, 12));
        plt.show();
    
    #时间序列,长期，季度及残差图像 
    def draw_time_series(self):
        decomposition = seasonal_decompose(self.time_series);
        trend = decomposition.trend;
        seasonal = decomposition.seasonal;
        resid = decomposition.resid;
        plt.subplot(411);
        plt.plot(self.time_series,label='Original');
        plt.legend(loc='best')
        plt.subplot(412);
        plt.plot(trend,label='trend');
        plt.legend(loc='best')
        plt.subplot(413);
        plt.plot(seasonal,label='seasonal');
        plt.legend(loc='best')
        plt.subplot(414);
        plt.plot(resid,label='resid');
        plt.legend(loc='best')
        plt.show();
        
    #单位根法检验
    def get_adf(self):
        print ('---------------Result of Dickry-Fuller test-------------------')
        dftest = adfuller(self.time_series);
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical value(%s)' % key] = value
        print (dfoutput)
        
    #获取AIC和BIC结果
    def get_AIC_BIC(self):
        print('-----------AIC AND BIC VALUE------------');
        print('AIC', self.aic);
        print('BIC', self.bic);
        
    #添加新数据
    def add_new_data(self,data,type='month'):
        if type=='day':
            new_index = self.time_series.index[-1]+relativedelta(days=1);
        elif type=='month':
            new_index = self.time_series.index[-1]+relativedelta(months=1);
        self.time_series[new_index] = data;

    #预测下一次值
    def forecast_next_data(self,type='month',d=0):
        if type=='day':
            end_index = self.time_series.index[-1]+relativedelta(days=1);
        elif type=='month':
            end_index = self.time_series.index[-1]+relativedelta(months=1);
        if (self.bic['p']==self.aic['p']) and (self.bic['q']==self.aic['q']):
            forecast_result = self.proper_model.predict(end=end_index,dynamic=True);
        else:
            print("aic'p and q is not eqauls bic's p and q");
        print('----------------------------------------------------------');
        print('the next data is ',self.get_reduction_value(forecast_result,d)[-1]);
    
    #检验模型性能,计算方差
    def get_RMSE(self):
        self.RMSE_data= self.time_series[self.predict_ts.index]  # 过滤没有预测的记录
        plt.figure(facecolor='white')
        self.predict_ts.plot(color='blue', label='Predict')
        self.RMSE_data.plot(color='red', label='Original')
        plt.legend(loc='best')
        plt.title('RMSE: %.4f'% np.sqrt(sum((self.predict_ts-self.RMSE_data)**2)/self.RMSE_data.size))
        plt.show();
    
    #设置time_series
    def set_time_series(self,time_series):
        self.time_series = time_series;
        
    #还原差分
    def get_reduction_value(self,data,d=0):
        if d==0:
            return data;
        else:
            diff_shift_ts = self.time_series.shift(d);
            diff_recover = time_series.add(diff_shift_ts);
        return diff_recover;
    
if __name__ == '__main__':
    #日期转换
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m');
    #读取数据
    cvs_file = 'C:\\Users\\wangkiing\Desktop\\人工智能\\51101_data.csv';
    #读入数据
    csv_series = pd.read_csv(cvs_file,encoding='gbk',index_col='年月',parse_dates=['年月'],date_parser=dateparse);
    time_series = pd.Series(csv_series['energy\r售电量']);
    time_series.index = pd.to_datetime(time_series.index) # 将字符串索引转换成时间索引
    model = ARIMA_Model(time_series);
    #观察时间序列图像及残差
    model.draw_time_series();
    #观察自相关性和偏相关性
    model.draw_acf_pacf();
    #观察残差信息
#    model.draw_resid_qq();
    #进行单位根-ADF检验
    model.get_adf();
    #获取aic和bic值
    model.get_proper_model();
    model.get_AIC_BIC();
   #处理数据，查看数据是否平稳，若不平稳，则进行差分,直至平稳为止
   #检验模型性能,计算方差
    model.get_RMSE();
   #预测下一次的值
    model.forecast_next_data();
    
    #---------------------------做一次差分后预测--------------
    time_series_diff_1=time_series.diff(1);
    #去除NA值
    time_series_diff_1.dropna(inplace=True);
    model.set_time_series(time_series_diff_1);
    #观察时间序列图像及残差
    model.draw_time_series();
    #观察自相关性和偏相关性
    model.draw_acf_pacf();
    #进行单位根-ADF检验
    model.get_adf();
    #获取aic和bic值
    model.get_proper_model();
    model.get_AIC_BIC();
    #检验模型性能,计算方差
    model.get_RMSE();
    #预测下一次的值
    model.forecast_next_data(d=1);