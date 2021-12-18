# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 10:04:09 2021

@author: kyau
"""
import pandas as pd
new_df=pd.DataFrame()
df=pd.read_csv('C:/Users/kyau/Desktop/data.csv')
df=df.set_index(['date','ticker'])#.unstack().head()
df=df.drop('volume',axis=1)
df.head()
df=df.sort_index()
df_clean=df.unstack()
#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
def get_w(df): #date as columns
    return (df-df.mean())/abs(df-df.mean()).sum()

def Z_score(df): #date as columns
    return (df-df.mean())/df.std()
        
summary_df=pd.DataFrame()
for pc in [40]:
    pipe=Pipeline([('dimension reduction', PCA(pc)), ('regressor', LinearRegression(n_jobs=-1))])
    ret_df=df_clean.pct_change().fillna(0)
    pred_df=pd.DataFrame()
    train_d=500
    for i in range(1500):
        train=(ret_df[i:i+train_d])
        pred=ret_df[i+train_d-1:i+train_d]
        pipe.fit(train,train)
        prediction=pipe.predict(pred)
        pred_df[pred.index[0]]=np.array(prediction).reshape(ret_df.shape[1],)
    pred_df=pred_df.set_index(train.columns).T
    signal=(np.tanh(Z_score(pred_df.T))-np.tanh(Z_score(ret_df.T)))
    signal.dropna(how='all',axis=1,inplace=True)
    w=get_w(signal).T.ewm(1).mean()
    index_ret=ret_df.dropna(how='all').dropna(how='all',axis=1).sum(axis=1)
    print('index sharpe:',(index_ret.mean()*252-2)/(index_ret.std()*np.sqrt(252)))
    ret=(w.shift(1)*ret_df).dropna(how='all').dropna(how='all',axis=1)
    ret_sum=ret.sum(axis=1)
    sh=(ret_sum.mean()*252)/(ret_sum.std()*np.sqrt(252))
    print('strategy sharpe:',sh)
    (ret_sum+1).cumprod().plot()
    sk=ret_sum.skew()
    turn=abs(w-w.shift(1)).sum(axis=1).mean()
    summary_df[pc]=[sh,sk,turn,ret_sum.mean()*25200]
    summary_df.index=['Sharpe','skew','turnover','ER']
    summary_df
summary_df.T.to_csv('C:/Users/kyau/Desktop/In_sample_summary_pc_40.csv')
summary_df.T

from statsmodels.tsa.stattools import adfuller as adf
for i in range(200):
    print(adf(signal.iloc[i,:]))

