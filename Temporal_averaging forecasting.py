# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:48:21 2023

@author: kosta
"""

"Mulptiple temporal averaging on MRM prices"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sktime.forecasting.arima import AutoARIMA
from sklearn.neural_network import MLPRegressor
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

path = "C:\\Users\kosta\\Documents\\PhD related\\Python\\Feature forecasting\\All metals.xlsx"

df = pd.read_excel(path,parse_dates=['DATE'],index_col='DATE')
df.head()

df = df.iloc[:,8].to_frame()
df.head()


def temporal_averaging(dataframe,periods_to_average):
    df2 = dataframe.copy()
    
    #df1 = df2.copy()
    for period in periods_to_average:
        deleted = 0
        df1 = df2.iloc[:,0].to_frame().copy()
        while len(df1)%period != 0:
                df1 = df1.drop(df1.index[0])
                deleted +=1 
                
        df2[str(period)+"-month"+" averaged"] = np.nan
        for i in range(len(df1),0,-period):
            df2[str(period)+"-month"+" averaged"].iloc[i+deleted-1] = df1.iloc[i-period:i].mean()
            
        

        
    return df2
    


dataset = temporal_averaging(df,[2,5,7,12])

plt.plot(dataset.iloc[:,0],color='black',label='Aluminum')
for j,i in zip(np.linspace(0.5,0.3,len(dataset.columns[1:])),dataset.columns[1:]):
    plt.plot(dataset[i].dropna(),alpha=j,color='red',label=i)
plt.legend()
plt.show()


def get_naive_forecasts(data,horizon):
    hierarchies = data.shape[1]
    
    forecasts = np.ones((horizon,hierarchies))
    
    for i in range(len(forecasts)):
        forecasts[i,:] = data.iloc[-1,:].values 
        
    return forecasts


def get_linreg_forecasts(data,horizon):
    # = data.shape[1]
    
    forecasts = pd.DataFrame(columns=data.columns)
    model = LinearRegression()
    for column in data.columns:
        k = data[column].dropna()
        k.index = pd.PeriodIndex(k.index,freq='M')
        model.fit(np.arange(len(k)).reshape(-1,1),k.values)
        preds = model.predict(np.array([len(k)]).reshape(-1,1))
        forecasts[column] = preds        
    return forecasts





def get_autoarima_forecasts(data,horizon):
    hierarchies = data.shape[1]
    forecasts = np.zeros((horizon,hierarchies))
    
    forecasts = pd.DataFrame(forecasts,columns=data.columns)
    model = AutoARIMA()
    for column in data.columns:
        k = data[column].dropna()
        k.index = pd.PeriodIndex(k.index,freq='M')
        model.fit(k.values)
        preds = model.predict(np.array([1]))
        for i in range(horizon):
            forecasts[column].iloc[i] = preds.flatten()       
    return forecasts



get_autoarima_forecasts(dataset, 1)


get_linreg_forecasts(dataset, 1)


f = get_naive_forecasts(dataset, 6)

n_val = 100
f = []
date_range = []
naive = []
for i in range(n_val,0,-1):
    df3 = df.iloc[:-i,]
    naive.append(df.iloc[-i-1].values)
    #df2 = temporal(df3,[2,3])
    df2 = temporal_averaging(df3,list(range(2,5)))
    f.append(np.mean(get_naive_forecasts(df2, 1),axis=1))
    date_range.append(pd.date_range(df3.index[-1]+pd.DateOffset(months=1),periods=1,freq='MS'))


f = np.array(f).flatten()
naive = np.array(naive).flatten()
plt.plot(date_range,f,'red',marker='o')
plt.plot(df.iloc[-n_val:],'black',marker='o')

print(f'RMSE naive : {np.sqrt(mean_squared_error(df.iloc[-n_val:,], naive))}')
print(f'RMSE tempavg : {np.sqrt(mean_squared_error(df.iloc[-n_val:,], f))}')

#### Lin Reg

n_val = 100
f = []
date_range = []
naive = []
for i in range(n_val,0,-1):
    df3 = df.iloc[:-i,]
    naive.append(df.iloc[-i-1].values)
    df2 = temporal_averaging(df3,range(2,3))
    #df2 = temporal(df3,list(range(2,5)))
    f.append(np.mean(get_linreg_forecasts(df2, 1).values,axis=1))
    date_range.append(pd.date_range(df3.index[-1]+pd.DateOffset(months=1),periods=1,freq='MS'))


f = np.array(f).flatten()
naive = np.array(naive).flatten()
plt.plot(date_range,f,'red',marker='o')
plt.plot(df,'black',marker='o')

print(f'RMSE naive : {np.sqrt(mean_squared_error(df.iloc[-n_val:,], naive))}')
print(f'RMSE tempavg : {np.sqrt(mean_squared_error(df.iloc[-n_val:,], f))}')



## AutoARIMA

n_val = 100
f = []
date_range = []
naive = []
arima = []
for1 = []
for i in range(n_val,0,-1):
    df3 = df.iloc[:-i,]
    naive.append(df.iloc[-i-1].values)
    #df2 = temporal(df3,range(2,6))
    df2 = temporal_averaging(df3,[12,24,36,48,60,72])
    forc= get_autoarima_forecasts(df2, 1)
    for1.append(forc.iloc[0,0])
    f.append(np.mean(forc.values,axis=1))
    date_range.append(pd.date_range(df3.index[-1]+pd.DateOffset(months=1),periods=1,freq='MS'))


f = np.array(f).flatten()
for1 = np.array(for1).flatten()
naive = np.array(naive).flatten()
plt.plot(date_range,f,'red',marker='o',markersize=2,label='tempavg_autoarima')
plt.plot(date_range,for1,'green',marker='o',markersize=2,label='autoarima')
plt.plot(df.iloc[-n_val:],'black',marker='o',markersize=2,label='real')
plt.legend()
plt.show()

print(f'RMSE naive : {np.sqrt(mean_squared_error(df.iloc[-n_val:,], naive))}')
print(f'RMSE tempavg : {np.sqrt(mean_squared_error(df.iloc[-n_val:,], f))}')
print(f'RMSE autoarima : {np.sqrt(mean_squared_error(df.iloc[-n_val:,], for1))}')



###################################### Neural Nets #####################################

dataset = temporal_averaging(df,[2,5,7,12])
X= dataset.iloc[:,0].to_frame().shift(1).dropna()

dataset


#dataset = dataset.fillna(method='ffill')
dataset = dataset.fillna(method='bfill')

dataset.dropna() 

#dataset = dataset.interpolate()

dataset.plot()
dataset

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=.2)

model = make_pipeline(MLPRegressor())

model.fit(X_train,y_train)

preds = model.predict(X_test)

plt.plot(y_test.index,preds)
plt.plot(y_test)







###################################### temporal_segregation #####################################

# Αντί να παίρνω την μέση τιμή μη Overlapping διαστημάτων παίρνω την τιμή ανά δίμηνο,τρίμηνο κοκ



def temporal_segregation(dataframe,segregates):
    dataf = dataframe.copy()
    for seg in segregates:
        dataf['seg' + str(seg)] = dataf.iloc[::seg,0]
        
    return dataf


temporal_segregation(df, [2,4,6])


n_val = 100
f = []
date_range = []
naive = []
arima = []
for1 = []
for i in range(n_val,0,-1):
    df3 = df.iloc[:-i,]
    naive.append(df.iloc[-i-1].values)
    #df2 = temporal(df3,range(2,6))
    df2 = temporal_segregation(df3,[2])
    forc= get_autoarima_forecasts(df2, 1)
    for1.append(forc.iloc[0,0])
    f.append(np.mean(forc.values,axis=1))
    date_range.append(pd.date_range(df3.index[-1]+pd.DateOffset(months=1),periods=1,freq='MS'))


f = np.array(f).flatten()
for1 = np.array(for1).flatten()
naive = np.array(naive).flatten()
plt.plot(date_range,f,'red',marker='o',markersize=2,label='tempavg_autoarima')
plt.plot(date_range,for1,'green',marker='o',markersize=2,label='autoarima')
plt.plot(df.iloc[-n_val:],'black',marker='o',markersize=2,label='real')
plt.legend()
plt.show()

print(f'RMSE naive : {np.sqrt(mean_squared_error(df.iloc[-n_val:,], naive))}')
print(f'RMSE tempavg : {np.sqrt(mean_squared_error(df.iloc[-n_val:,], f))}')
print(f'RMSE autoarima : {np.sqrt(mean_squared_error(df.iloc[-n_val:,], for1))}')




