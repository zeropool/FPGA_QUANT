# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:46:23 2018

@author: Max Fang
"""

import numpy as np
import pandas as pd
from ini import *
from numba import jit
import matplotlib.pyplot as plt

def QPData(colname):
    df = pd.read_csv(path+colname+'.csv',index_col = 0)
    return df.iloc[(df.index >= start_date)&(df.index<=end_date),:-3]
def get_trd_date():
    df = pd.read_csv(path + 'dates.i.csv',index_col=0)
    return df.loc[(df.x>=start_date) & (df.x <= end_date),:]
def get_weight(df):
    df = df.shift(1)
    df.iloc[-1,:] = df.iloc[-1,:] * 0.0
    df = df.apply(lambda x:normalize_weight(x),axis=1)
    return df

def normalize_weight(arr):
    arr[~np.isfinite(arr)] = 0.0
    pos = arr[arr>0].sum()
    neg = arr[arr<0].sum()
    res = np.where(arr>0,arr/pos,arr/abs(neg))
    return res

def drawdown(v):
    l=len(v)
    cs = v.cumsum()
    return np.array([cs[i]-cs[i:].min() for i in range(l)]).max()
           
def sharpe(v):
    l=len(v)
    cs= v.cumsum()
    return cs.mean()/cs.std()

def rank(df):
    return df.rank(axis=1)

def ts_rank(df,n):
    return df.rolling(n).apply(lambda x:pd.DataFrame(x).rank(axis=0).values[-1,:])

def ts_max(df,n):
    return df.rolling(n).apply(lambda x:x.max(axis=0))
def delay(df,n):
    return df.shift(n,axis=0)

def sum1(df,n):
    return df.rolling(n,axis=0).sum()
def stddev(df,n):
    return df.rolling(n,axis=0).std()
def decay_linear(df,n):
    wt = np.arange(1,n+1)
    wt = wt/wt.sum()
    return df.rolling(n,axis=0).apply(lambda x:wt.dot(x))
def ts_argmax(df,n):
    return df.rolling(n).apply(lambda x:n-x.argmax(axis=0)*1.0)
def signedpower(df1,df2):
    return df1**df2
def delta(df,n):
    return df - df.shift(n,axis=0)
def optional(booldf,df1,df2):
    return pd.DataFrame(np.where(booldf,df1,df2),index = df1.index,columns = df1.columns)
def center_rank(df):
    return df.apply(lambda x:Center_Rank(x),axis=1)  
def Center_Rank(arr):
    v=arr.copy()
    v[~np.isfinite(v)]=np.nan
    r=pd.Series(v).rank().values*1.0/(len(v)-np.isnan(v).sum())
    return r-np.nanmean(r)
    
#####################################################################
def covariance(df1,df2,d):
    m1 = df1.values
    m2 = df2.values
    df = pd.DataFrame(covariance_rolling(m1,m2,d),index = df1.index,columns = df1.columns)
    return df
@jit
def covariance_rolling(m1,m2,d):
    m,n = m1.shape
    res = m1 * np.nan
    for i in range(d,m+1):
        for j in range(n):
            res[i-1,j] = cov(m1[(i-d):i,j],m2[(i-d):i,j])
    return res
@jit
def cov(arr1,arr2):
    idx = np.isfinite(arr1) & np.isfinite(arr2)
    s1,s2 = arr1[idx],arr2[idx]
    return np.cov(s1,s2)[0,1]
######################################################################
def correlation(df1,df2,d):
    m1 = df1.values
    m2 = df2.values
    df = pd.DataFrame(correlation_rolling(m1,m2,d),index = df1.index,columns = df1.columns)
    return df
@jit
def correlation_rolling(m1,m2,d):
    m,n = m1.shape
    res = m1 * np.nan
    for i in range(d,m+1):
        for j in range(n):
            res[i-1,j] = corr(m1[(i-d):i,j],m2[(i-d):i,j])
    return res
@jit
def corr(arr1,arr2):
    idx = np.isfinite(arr1) & np.isfinite(arr2)
    s1,s2 = arr1[idx],arr2[idx]
    return np.corrcoef(s1,s2)[0,1]
######################################################################


def standardlize(df):
    std = df.std(axis=1)
    mean = df.mean(axis=1)
    return ((df.T - mean)/std).T

def winsorize(df):
    return df.apply(lambda x:maxregularize(x),axis=1)

def maxregularize(arr):
    Series = pd.Series(arr)
    per95=Series.quantile(0.975,interpolation='nearest')
    per05=Series.quantile(0.025,interpolation='nearest')
    Series.loc[Series>per95]=per95
    Series.loc[Series<per05]=per05
    factor=2*(Series-per05)/(per95-per05)-1
    return factor

def ret_in_n_days(close_path,n):
    c = pd.read_csv(close_path,index_col = 0)
    return (c.shift(-n,axis=0))/c - 1