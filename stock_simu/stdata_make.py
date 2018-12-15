# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:29:46 2018

@author: Max Fang
"""

import tushare as ts 
import pandas as pd
import numpy as np

path = 'C:/cygwin64/home/MaxFang/Quant/stdata/'


df = ts.get_today_all()
a_shares = (np.floor(df.code.astype('int')/1e5)<=6.0) #A股以0,3,6开头，部分b股以9开头
all_ticker = df.code[a_shares]
all_ticker = all_ticker[~all_ticker.duplicated()]
ticker_chinese = df.name[a_shares]


c = pd.DataFrame()
i = 0
for ticker in all_ticker:
    i += 1 
    print(ticker,i)
    df = ts.get_k_data(ticker)
    c = pd.concat((c,df))

c.to_csv('all_stdata.csv')

def pivot(value):
    return pd.pivot(c.date,c.code,value).sort_index()

close = pivot(c.close)
close = close.fillna(method ='ffill')
close = close.loc[close.index>'2016-01-01',:]
close.to_csv('close.csv')

def merge(old_close,new_close):
    
    return latest_adjusted_close


    

        
        
        
        