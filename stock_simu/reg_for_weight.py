# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:19:17 2018

@author: Max Fang
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn import svm
from sklearn import metrics
import seaborn as sns
from reg_function import *
#model = LinearRegression()

fund_list = ['pb','pettm','roattm','npgrttm']
#alpha_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
alpha_list = [8,13,16]
fund_fname = ['{}.csv'.format(i) for i in fund_list]
alpha_fname = ['Alpha{}_new.csv'.format(i) for i in alpha_list]
fund_varname = fund_list.copy()
alpha_varname = ['Alpha{}'.format(i) for i in alpha_list]
#factor_varname = fund_varname + alpha_varname
factor_varname = alpha_varname
#factor_varname = fund_varname
# read fundmental factors
for i in range(len(fund_fname)):
    locals()[fund_varname[i]] = get_std_data(fund_fname[i])
    print('reading ',fund_fname[i],"finished!")

#read alpha factors
for i in range(len(alpha_fname)):
    locals()[alpha_varname[i]] = get_std_data(alpha_fname[i])
    print('reading ',alpha_fname[i],"finished!")
    
c = pd.read_csv('close.csv',index_col = 0)
c.index = pd.to_datetime(c.index,format = '%Y%m%d')
tickers =list(c.columns)
csi300 = pd.read_csv('csi300.csv',index_col = 0)
csi300.index = pd.to_datetime(csi300.index,format = '%Y%m%d')
ret30 = pd.read_csv('ret30.csv',index_col = 0)
ret30.index = pd.to_datetime(ret30.index,format = '%Y%m%d')
ret = pd.read_csv('ret30.csv',index_col = 0)
ret.index = pd.to_datetime(ret.index,format = '%Y%m%d')

## all trade date | trade_date | trade_points
date_list = pd.read_csv('date.csv',index_col = 0,names = ['dates'])
trade_dates = pd.read_csv('trade_dates.csv',index_col = 0,names=['trade_dates'])
trade_dates = list(trade_dates.values.flatten())
trade_points = pd.read_csv('trade_points.csv',index_col =0,names=['trade_points'])
wt_matrix = pd.DataFrame(1,index = Alpha103.index ,columns = factor_varname)

for i in range(40,(len(trade_dates)-1)):
    yesterday = trade_dates[i-1]
    today = trade_dates[i]
    print('today:',today)
    next_1_day =  trade_dates[i+1]
    next_2_day = trade_dates[i+2]
    csi300_ind = csi300.loc[today,:]
    csi300_tickers = list(csi300_ind[csi300_ind].index)
    y_next = ret.loc[next_1_day,csi300_tickers]
    y_next2 = ret.loc[next_2_day,csi300_tickers]
    X = pd.DataFrame()
    X1 = pd.DataFrame()
    X['ret'] = y_next
    X1['ret'] = y_next2
    for j in (factor_varname):
        alpha = locals()[j].copy()
        X[j] = alpha.loc[today,csi300_tickers] #- alpha.loc[yesterday,csi300_tickers]
        X1[j] = alpha.loc[next_1_day,csi300_tickers]
    X = X.dropna(axis=0,how='any')
    X1 = X1.dropna(axis=0,how='any')
    X_next = X1.iloc[:,1:]
    Y_next2 = X1.iloc[:,0]
    X_train = X.iloc[:,1:]
    Y_train = X.iloc[:,0]
    
'''
Linear Regression
'''
# =============================================================================
#     wt1 = np.linalg.inv((X_train.T.dot(X_train))).dot(X_train.T).dot(Y_train)
#     Y_learn = X_train.dot(wt1)
#     error = (Y_train - Y_learn)
#     sns.distplot(error,color = 'g')
#     coef = np.corrcoef(Y_train,Y_learn)[0,1]
#     print("coef : ",coef)
#     index1 = trade_dates.index(today)
#     index2 = trade_dates.index(next_month_day)
#     wt_matrix.loc[(wt_matrix.index>today)&(wt_matrix.index<=next_month_day),:] \
#     = np.repeat(wt1.reshape((1,len(wt1))),repeats =(index2-index1),axis=0)
# =============================================================================

'''
组合因子过程
'''
# =============================================================================
# m,n = Alpha103.shape
# combine = 0.0 * Alpha103
# for i in range(m):
#     for j,varname in enumerate(factor_varname):
#         combine.iloc[i,:] = wt_matrix.iloc[i,j] * locals()[varname].iloc[i,:]  
# =============================================================================
    
 '''
 Adaboost Regression
 '''   
# =============================================================================
#     rng = np.random.RandomState(1)
#     model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)
#     model.fit(X_train,Y_train)
# #    Y_learn = model.predict(X_train)
#     Y_next_learn = model.predict(X_next)
#     coef = np.corrcoef(Y_next2,Y_next_learn)[0,1]
#     print("coef : ",coef)
# =============================================================================

'''
测试Adaboost的过拟合能力
'''

# =============================================================================
# Y_train = np.random.rand(100)
# X_train = np.random.rand(100,5)
# rng = np.random.RandomState(1)
# model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)
# model.fit(X_train,Y_train)
# Y_learn = model.predict(X_train)
# coef = np.corrcoef(Y_train,Y_learn)[0,1]
# print("coef : ",coef)
# =============================================================================



    
        