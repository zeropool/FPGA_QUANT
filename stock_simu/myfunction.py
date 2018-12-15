#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:55:53 2018

@author: fangzhong
"""

import numpy as np
import pandas as pd
import re
from numba import jit
from qp import *
import matplotlib.pyplot as plt
def Stdev(v):
    l = len(v)
    return v.std() * np.sqrt(l/(l-1))

def ufunc_std(arr):
    return arr.std()

@jit
def numba_std(arr):
    mean=0.0
    var=0.0
    for i in arr:
        mean += i/len(arr)
    for i in arr:
        var  += (i-mean)**2
    return var**0.5/(len(arr)-1)
@jit
def numba_mean(matrix):
    row,col=matrix.shape
    mean=np.zeros(row)
    for i in range(row):
        for j in range(col):
            mean[i] += matrix[i,j]/col
    return mean
    
def Rank(v):
    order = v.argsort(kind='mergesort')
    r = (order.argsort(kind='mergesort') +1)/len(v)
    return r-r.mean()
#    r=pd.Series(v).rank().values/len(v)
    
#    return pd.Series(v).rank().values
@jit    
def TSRank(matrix):
    m=matrix.copy()
    m[~np.isfinite(m)]=np.nan
    return pd.DataFrame(m).rank(axis=1).values[:,-1]*1.0

def TSArgMax(matrix):
    m=matrix.copy()
    return m.shape[1]-m.argmax(axis=1)*1.0

#def TSArgMax(m, d):
#    nr,nc=m.shape
#    eidx=nc
#    sidx = 0 if eidx < d else eidx-d
#    return eidx - sidx -1 -m[:,sidx:eidx].argmax(axis=1)

def Corrwith(m1,m2):
    return pd.DataFrame(m1).corrwith(pd.DataFrame(m2),axis=1).values

#def critical_pnl_corrwith(pnl_m1,pnl_m2):
#    ts1=pnl_m1.sum(axis=1)
#    ts2=pnl_m2.sum(axis=1)
#    top140=
    


def alphacorr(alpha1,alpha2):
    index=pd.Series(all_instruments()).isin(pd.Series(tradable_instruments()))
    return Corrwith((alpha1[:,1:])[:,index],(alpha2[:,1:])[:,index]).mean()

def TSArgMin(m,d):
    nr,nc=m.shape
    eidx=nc
    sidx=0 if eidx <d else eidx-d
    return eidx -sidx -1 -m[:,sidx:eidx].argmin(axis=1)    
def ZScore(v):
    m = v.mean()
    s = Stdev(v)
    return (v-m)/s if s>0 else (v-m)
@jit
def Center_Rank(arr):
    v=arr.copy()
    v[~np.isfinite(v)]=np.nan
    r=pd.Series(v).rank().values*1.0/(len(v)-np.isnan(v).sum())
    return r-np.nanmean(r)

def crw(v,wt):
    arr=regularize(v*wt)
    mean=np.nanmean(arr)
    arr=arr-mean
    arr[arr>0]=arr[arr>0]/np.nansum(arr[arr>0])
    arr[arr<=0]=arr[arr<=0]/abs(np.nansum(arr[arr<=0]))
    return arr
    

def scale(v):
    return v/np.nansum(v)

@jit
def decay_linear(m):
    d=m.shape[1]
    wt=np.arange(1,d+1)/np.arange(1,d+1).sum()
    return m.dot(wt)

def delay_all(QPData,d):
    return QPData.history_matrix_all(d+1)[:,0].flatten()
def delay(QPData,d):
    return QPData.history_matrix(d+1)[:,0].flatten()
def todf(long,short,summarystr):
    pattern=re.compile(r'\-?[\w\/]+\.?\d?\d?')
    result=pattern.findall(summarystr)
    arr=np.array(result).reshape(int(len(result)/13),13)
    arr_temp=np.array(['long','short']+[long,short]*(int(len(result)/13)-1))
    arr1=arr_temp.reshape(int(len(arr_temp)/2),2)
    arr2=np.hstack([arr1,arr])
    df=pd.DataFrame(arr2[1:],columns=arr2[0])
    return df
def get_factor_name(ini_path):
    ls=[]
    pattern = re.compile(r'(\w+) =')
    with open(ini_path,'r') as f:
        for line in f.readlines():
            line=line.strip()
            if ('#' in line) or ('=' not in line) or ('[' not in line) or (']' not in line):
                continue
            else:
                line=line.split('=',1)[0]
                line=line.strip()
                ls.append(line)
    return ls
@jit
def schmidt(alpha_matrix):
    orth=0.0*alpha_matrix
    n=alpha_matrix.shape[0]
    m=alpha_matrix.shape[1]
    orth[:,0]=alpha_matrix[:,0]
    for i in range(1,m):
        cross=np.zeros(n)
        for j in range(i):
            cross=cross+orth[:,j].dot(alpha_matrix[:,i])/orth[:,j].dot(orth[:,j])*orth[:,j]
        orth[:,i]=alpha_matrix[:,i]-cross
    for k in range(m):
        orth[:,k]=orth[:,k]/(orth[:,k].dot(orth[:,k]))**0.5
    return orth
    
def split_index(close_all,string):
    arr=close_all[-4:,:].T
    csi500=arr[:,0]
    csi300=arr[:,1]
    sse180=arr[:,2]
    sse50=arr[:,3]
    return pd.DataFrame(eval(string))

def orthogonal(v1,v2):
    return v1-np.nansum(v1*v2)/np.nansum(v2*v2)*v2


def plot_index(csi):
    csi=csi.values
    value=csi[(csi==csi)&(~np.isinf(csi))]
    b=np.linspace(-0.1,0.1,21)
    l=list(range(-10,0))+list(range(1,11))
    res=pd.cut(value,bins=b,labels=l).describe()
    plt.bar(res.index,res.counts)
    plt.bar(res.index,res.freqs)
    return res

def pairs(stock1,stock2):
    idx1=get_instrument_index(stock1)
    print(get_id(idx1))
    idx2=get_instrument_index(stock2)
    print(get_id(idx2))    
    close=QPData('adjusted_close').history_matrix_all(get_date_index(today)\
                 -get_date_index(20070104)+1,Constants.Now)
    close[np.where(close)==0.0]=np.nan
#    coef_mat=pd.DataFrame(close.T).corr().values
    
    c_stock1=pd.Series(close[idx1,:])
    c_stock2=pd.Series(close[idx2,:])    
    return pd.Series(c_stock1).corr(c_stock2)#,coef_mat    
def shares_df(simu):
    tickers=all_instruments()
    trade_dates=all_dates()[get_date_index(20100104):get_date_index(20151231)+1]
    share_df=pd.DataFrame(simu.info(field='shares')[:,:,0],index=trade_dates,\
                          columns=tickers).T
    return share_df
def prices(simu):
    tickers=all_instruments()
    trade_dates=all_dates()[get_date_index(20100104):get_date_index(20151231)+1]
    share_df=pd.DataFrame(simu.info(field='prices')[:,:,0],index=trade_dates,\
                          columns=tickers).T
    return share_df

def ordered_df(simu):
    tickers=all_instruments()
    trade_dates=all_dates()[get_date_index(20100104):get_date_index(20151231)+1]
    share_df=pd.DataFrame(simu.info(field='shares')[:,:,0],index=trade_dates,\
                          columns=tickers).T
    return share_df
   
def fill(simu):
    tickers=all_instruments()
    trade_dates=all_dates()[get_date_index(20100104):get_date_index(20151231)+1]
    share_df=pd.DataFrame(simu.info(field='filled')[:,:,0],index=trade_dates,\
                          columns=tickers).T
    return share_df
def gmv_long_short(simu):
    prs=prices(simu)
    share=shares_df(simu)
    gmv=prs*share
    df=pd.DataFrame()
    df['long_gmv']=gmv.apply(lambda x:x[x>0].sum(),axis=0)
    df['short_gmv']=gmv.apply(lambda x:x[x<0].sum(),axis=0)
    df['gmv']=np.abs(gmv).sum(axis=0)
    return df

def pnl_long_short(simu):
    shares=shares_df(simu).T # row:day; col:tickers 
    pnl_type_name=['Day','Night','Trade','TCost','Fees','Dividend']
    pnl=0.0*shares
    for pnl_type in pnl_type_name:
        df=simu.pnl_detail_by_type(pnl_type)
        pnl = pnl + df
    long_pnl=pnl[shares>0].sum(axis=1)
    short_pnl=pnl[shares<0].sum(axis=1)
    all_pnl=pnl.sum(axis=1)            
    return pd.DataFrame({'long_pnl':long_pnl,'short_pnl':short_pnl,'all_pnl':all_pnl})

def fields_onestock(sim,stock_str):
    field_name=['shares','ordered','filled','prices','tcost','fees']
    trade_dates=all_dates()[get_date_index(20100104):get_date_index(20151231)+1]
    field_info=pd.DataFrame(sim.info(stock=stock_str)[:,0,:],columns=field_name,\
                 index=trade_dates)
    field_info['All_pnl']=pnl_onestock_process(sim,stock_str)['All_pnl']
    return field_info
def pnl_gmv_ret(simu):
    port=simu.portfolio
    pnl=port.daily_pnl
    gmv=port.daily_gmv
    ret=port.daily_ret
    trade_dates=all_dates()[get_date_index(20100104):get_date_index(20151231)+1]
    pnl_gmv_ret=pd.DataFrame({'pnl':pnl,'gmv':gmv,'ret':ret},index=trade_dates)
    return pnl_gmv_ret
def pnl_oneday(sim,date_int):
    return sim.pnl_detail_by_date(date_int).sum(axis=1).sort_values(ascending =False)
def pnl_onestock_process(sim,stock_str):
    df=sim.pnl_detail_by_id(stock_str)
    df['All_pnl']=df.sum(axis=1)
    return df

def outdir():
    return get_ini().findString("OutputDir")

def show_fac_effic():
    fac_exposures=readm2df(outdir() + "/factor_risk.bin")
    fac_ex =fac_exposures.mean(axis = 0)   
    
    fac_pnl=readm2df(outdir() + "/factor_pnl.bin")
    fac_pnl =fac_pnl.sum(axis = 0) 
    fac_pnl_per = np.round(fac_pnl/fac_pnl.sum()*100, 1)
    
    pd_list = [pd.DataFrame(fac_ex,columns = ['factor_expo']),pd.DataFrame(fac_pnl,columns = ['factor_pnl'])]
    ind_pd = pd.concat(pd_list,axis = 1)
    ind_pd.sort_values('factor_pnl',ascending = False, inplace = True)
    plt.rcParams['figure.figsize']=(12,10)
    ind_pd.plot(kind = 'bar', subplots = True, figsize = (12,10))
    plt.ylabel("Pnl (Wan) RMB")
    plt.savefig(outdir() + '/fac.png')
    plt.show()



    
def sell_in_thursday(int_monday):
    Monday=20100104
    dates=pd.date_range(start=str(Monday+int_monday-1),end='20151231',freq='7D')
    return dates.strftime('%Y%m%d').astype('int')

def resample_dates(all_dates,frequency='M'):
    d=all_dates()
    trade_dates=d[(d>=20100104)&(d<=20180726)]
    ind=pd.to_datetime(trade_dates,format='%Y%m%d')
    trade_points=pd.Series(trade_dates,index=ind).resample(frequency).first().dropna()
    return trade_points.values.astype('int')

def regularize(v):
    per95=np.percentile(v,97.5)
    per05=np.percentile(v,2.5)
    v[v>per95]=per95
    v[v<per05]=per05
    return 2*(v-per05)/(per95-per05)-1

def destribution(string='volume.b5.bin'):
    m3d=readm(string)
    a=np.nanmean(m3d,axis=(0,1))
    plt.stem(a)
    return m3d
def myreadm(field,ticker='601939'):
    value=readm('/qp/data/platform_objects/core/'+field+'.bin')
    return value[:,get_instrument_index(ticker)]
def ind_factor_cate(indzx,factor):
    ind=np.floor(indzx/1000000)%100
    df=pd.DataFrame({'factor':factor,'ind':ind})
    df['factor_level']=df.factor.groupby(df.ind).transform(lambda x:np.where(x>=np.percentile(x,80),0.0,1.0))
    return df.factor_level.values
@jit
def ind_neutral(alpha_v,ind):
    alpha_v[~np.isfinite(alpha_v)]=np.nan
    ind_v=np.round(ind/1000000)%100
    s=pd.Series(alpha_v,index=ind_v)
    return s.groupby(s.index).transform(Center_Rank).values


@jit
def ind_factor(factor_all,ind_all):
    ind_all_1=np.floor(ind_all/1000000)%100 #Industry 1
    ind_all_2=np.floor(ind_all/10000)%100   #Industry 2
    ind_all_3=np.floor(ind_all/100)%100 # Industry 3
    ind_all_4=ind_all%100
    s1=pd.Series(factor_all,index=ind_all_1)
    return s1.groupby(s1.index).transform(sum).values
@jit
def seperate(v):
    ninty_pct=np.percentile(v,90)
    level_v=np.where(v>=ninty_pct,'big','small')
    return level_v

def ind_headidx(indzx,cap,ret):
    ind=np.floor(indzx/1000000)%100
    df=pd.DataFrame({'cap':cap,'ind':ind,'ret':ret})
    df['cap_level']=df.cap.groupby(df.ind).transform(lambda x:np.where(x>=np.percentile(x,90),'big','small'))
    df['std_sector']=df.ret.groupby(df.ind).transform(lambda x:np.nanstd(x))
    df['ret_mean_sector']=df.ret.groupby([df.ind,df.cap_level]).transform(lambda x:np.nanmean(x))
#    print(df.ret_mean_sector.groupby(df.ind).apply(lambda x:np.unique(x).shape))
    df['big_small']=df.ret_mean_sector.groupby(df.ind).\
    transform(lambda x:abs(np.unique(x)[0]-np.unique(x)[1]) if np.unique(x).shape[0]==2 else np.nan)
    df['head_index']=df.big_small/df.std_sector
    return df.head_index.values

def cap_pb_neutral(alpha,cap,pb):
    df=pd.DataFrame({'cap':cap,'pb':pb,'alpha':alpha})
    df['cap_level']=cate(cap,N=5)
    df['pb_level']=df.pb.groupby(df.cap_level).transform(lambda x:np.where(x>=np.percentile(x,90),'big','small'))
    df['cap_pb_neutral']=df.alpha.groupby([df.cap_level,df.pb_level]).transform(Center_Rank)
    return df.cap_pb_neutral.values
  
def cate(v,N='10'):
    series=regularize(v)
    bins=np.percentile(series,np.arange(0,100+100/N,100/N))
    bins[0]=-1.01
    cate_series=pd.cut(series,N,labels=range(1,N+1))
    return cate_series.get_values()*1.0
@jit
def regression(v1,v2):
    idx1=np.isfinite(v1)
    idx2=np.isfinite(v2)
    y=v1[idx1 & idx2]
    x=v2[idx1 & idx2]
    x=np.array([np.ones(len(x)),x])
    theta=np.linalg.inv(np.dot(x,x.T)).dot(x).dot(y.T)
    
    return theta, v1-theta[0]-theta[1]*v2
@jit
def multi_regression(v1,weight,*args,):
    idx1=np.isfinite(v1)
    idx2=idx1.copy()
    for v in args:
        idx=np.isfinite(v) 
        idx2=idx2 & idx
    y = v1[idx2]
    n = len(y)
    ls = [np.ones(n)]
    ls_ini = [np.ones(len(v1))]
    for v in args:
        ls_ini.append(v)
        ls.append(v[idx2])
    x_ini = np.array(ls_ini).T
    x = np.array(ls).T
    w=np.diag(weight[idx2])
    theta = np.linalg.inv(x.T.dot(w).dot(x)).dot(x.T).dot(w).dot(y.T)
    residual=(v1 - x_ini.dot(theta.T))
    return residual,theta
def regularize(arr):
    v=arr.copy()
    idx=np.isfinite(v)
    v[~idx]=np.nan
    s=v[idx]
    per95=np.percentile(s,95)
    per05=np.percentile(s,5)
    v[v>per95]=per95
    v[v<per05]=per05
    return (2*(v-per05)/(per95-per05)-1)

# in gokudata
def maxregularize(arr):
    Series = pd.Series(arr)
    per95=Series.quantile(0.975,interpolation='nearest')
    per05=Series.quantile(0.025,interpolation='nearest')
    Series.loc[Series>per95]=per95
    Series.loc[Series<per05]=per05
    factor=2*(Series-per05)/(per95-per05)-1
    return factor

def top(arr,pct):
    v=arr.copy()
    v[~np.isfinite(v)]=np.nan
    fi_len=np.isfinite(v).sum()
    N=fi_len*(1-2*pct)
    rk=pd.Series(v).rank().values
    res=np.where(((rk<=N/2.0)|(rk>=fi_len-N/2.0+1)),v,np.nan)
    return res
#    return QPData.get_all(Constants.Yesterday)-delay_all(QPData,d)
#def delta_new_all(QPData,d):
#    return QPData.get_all(Constants.Now)-delay_all(QPData,d-1)
#def delta(QPData,d):
#    return QPData.get_tradable(Constants.Yesterday)-delay(QPData,d)
#def delta_new(QPData,d):
#    return QPData.get_tradable(Constants.Now)-delay(QPData,d-1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    