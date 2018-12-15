# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:32:56 2018

@author: Max Fang
"""

from qp import *
import ini
#from myfunction import *
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#sys.path.append('../Quant/stdata/')
close = QPData('close')
uclose = QPData('uclose')
open1 =  QPData('open')
uopen = QPData('uopen')
high = QPData('high')
cap = QPData('cap')
low = QPData('low')
liq101 = QPData('liq101')
volume = QPData('volume')
uvolume = QPData('uvolume')
vwap = QPData('vwap')
estur = QPData('estu')
csi300 = QPData('csi300')
ind = QPData('indzx')
share_out = QPData('so')
adj = QPData('adj.factor')
'''
data_dictionary.xlsx内所有数据


'''
trade_dates = get_trd_date()
ind1 = np.floor(ind/1e6)%100
filled_price = open1.values # should be changed to price at 10:00

def Strategy():
    alpha = -close/vwap
    alpha = alpha.rolling(5).mean()
    alpha = alpha * estur
    wt = center_rank(alpha)
    '''
    1.根据仓位和停牌信息调整调仓方法
    2.对各种风险因子进行回归中性化，选择放开风格因子（承担适当风险）
    3.投资组合优化
    '''
    '''
    量价因子
    基本面因子
    资金流因子
    '''
#    wt = (rank(ts_argmax(signedpower(optional(returns<0,stddev(returns,20),close),2),5))-0.5)
#    turnover = volume/(share_out * adj)
#    wt = center_rank(-stddev(turnover,20))
#    wt = center_rank(-liq101)
    return wt

def backtest():
    alpha = Strategy()
    m,n = close.values.shape
    net_pnl = np.zeros((m,n))
    RMBTrd = np.zeros((m,n))
    shares = np.zeros((m,n))
    filled = np.zeros((m,n))
    ordered = np.zeros((m,n))
    wt = get_weight(alpha).values
    adj_shares = wt * Allocation_Fixed_Capital * 1e6 / filled_price
    adj_shares = np.floor(adj_shares/100)*100
    adj_shares[adj_shares!=adj_shares] = 0.0
    O = open1.values
    C = close.values
    VWAP = vwap.values
    for i in range(1,len(trade_dates)):
        print("today:",trade_dates.values[i])
        shares[i,:] = shares[i-1,:] + filled[i-1,:] # start of day shares
        ordered[i,:] = adj_shares[i,:] - shares[i,:]
        MaxVolume = volume.values[i,:]*MaxVolumePercent*0.01
        orderVolume = ordered[i,:]
        filled[i,:] = np.where(orderVolume<MaxVolume,orderVolume,MaxVolume) #buying(ordered,volume) filled shares
        night_pnl = (O[i,:] - C[i-1,:]) * shares[i,:]
        day_pnl = (C[i,:]-O[i,:]) * shares[i,:]
        trade_pnl = (C[i,:] - VWAP[i,:]*(1+BPSAboveVWAP*0.0001)) * filled[i,:]
        RMBTrd[i,:] = C[i,:]*(1+BPSAboveVWAP*0.0001)*abs(filled[i,:])/100
        fees      =  abs(VWAP[i,:]*(1+BPSAboveVWAP*0.0001) * filled[i,:]) * trd_fee 
        net_pnl[i,:] = night_pnl + day_pnl + trade_pnl -fees
    return net_pnl,RMBTrd,shares,filled,ordered,wt
res = backtest()  

def simplot(result):
    net_pnl,RMBTrd,shares,filled,ordered,wt = result
    daily_pnl  = np.nansum(net_pnl,axis =1)
    daily_trd = np.nansum(RMBTrd,axis=1)
    trd = pd.Series(daily_trd/1e6,index = close.index)
    trd.index = pd.to_datetime(trd.index,format="%Y%m%d") 
    debug_pnl =  pd.DataFrame(net_pnl,index = close.index,columns = close.columns)
    cum_pnl = daily_pnl.cumsum()
    se = pd.Series(daily_pnl/(Allocation_Fixed_Capital * 1e6) * 100,index = close.index)
    se.index = pd.to_datetime(se.index,format="%Y%m%d")
    cum_ret = se.cumsum()
    cum_ret.to_csv('cum_pnl.csv')
    date,pnl=np.loadtxt('cum_pnl.csv',delimiter=',',converters={0:mdates.bytespdate2num('%Y-%m-%d')},skiprows=1,usecols=(0,1),unpack=True)
    plt.plot_date(date,pnl,'-')
    plt.grid(True)
    plt.ylabel('%GMV')
    plt.xlabel('Date')
    return se,trd,RMBTrd,wt

se,trd,RMBTrd,wt = simplot(res)

df = pd.DataFrame()
df['Return'] = se.resample('a').sum().values
df['MaxDD'] = se.resample('a').apply(lambda x:drawdown(x)).values
df['Sharpe'] = se.resample('a').apply(lambda x:sharpe(x)).values
#df['RMBTrd'] = trd.resample('a').mean().values
df.index = np.arange(start_date//10000,end_date//10000 + 1)
df.loc['All'] = [se.sum(),drawdown(se),sharpe(se)]
print(df)



