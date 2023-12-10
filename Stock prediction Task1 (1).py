#!/usr/bin/env python
# coding: utf-8

# # # # BHARAT INTERN
# # 
# # # NAME- MURTHY TELAGAREDDY
# # 
# # # TASK1-STOCK PREDICTION
# #  - IN THIS WE WILL USE THE NSE TATA GLOBAL BEVERAGES DATASET FOR STOCK PREDICTION

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # NSE-TATAGLOBAL DATASETS

# # Stock Market Prediction And Forecasting Using Stacked LSTM
# 

# ### To build the stock price prediction model, we will use the NSE TATA GLOBAL dataset. This is a dataset of Tata Beverages from Tata Global Beverages Limited, National Stock Exchange of India: Tata Global Dataset

# ## Import Libraries

# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import pandas as pd
import io
import requests 
import datetime


# ## Import Datasets

# In[4]:


df = pd.read_csv('../input/new-tata-dataset/NSE-TATAGLOBAL.csv')
df.head()


# In[5]:


df1 = pd.read_csv('../input/new-tata-dataset/NSE-TATAGLOBAL.csv')
df1.head()


# ## Shape of data

# In[6]:


df.shape


# ## Gathering information about the data

# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.dtypes


# ## Data Cleaning

# ### Total percentage of data is missing

# In[10]:


missing_values_count = df.isnull().sum()

total_cells = np.product(df.shape)

total_missing = missing_values_count.sum()

percentage_missing = (total_missing/total_cells)*100

print(percentage_missing)


# In[11]:


NAN = [(c, df[c].isnull().mean()*100) for c in df]
NAN = pd.DataFrame(NAN, columns=['column_name', 'percentage'])
NAN


# # Data Visualisation

# In[12]:


sns.set(rc = {'figure.figsize': (20, 5)})
df['Open'].plot(linewidth = 1,color='blue')


# In[13]:


df.columns


# In[14]:


cols_plot = ['Open','High','Low','Last','Close']
axes = df[cols_plot].plot(alpha = 1, figsize=(20, 30), subplots = True)

for ax in axes:
    ax.set_ylabel('Variation')


# ## Sort the dataset on date time and filter “Date” and “Open” columns

# In[15]:


df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']
df


# In[16]:


del df["Date"]


# In[17]:


df.dtypes


# ## 7 day rolling mean

# In[18]:


df.rolling(7).mean().head(10)


# In[19]:


df['Open'].plot(figsize=(20,8),alpha = 1)
df.rolling(window=30).mean()['Close'].plot(alpha = 1)


# In[20]:


df['Close: 30 Day Mean'] = df['Close'].rolling(window=30).mean()
df[['Close','Close: 30 Day Mean']].plot(figsize=(20,8),alpha = 1)


# ## Optional specify a minimum numbe2of periods

# In[21]:


df['Close'].expanding(min_periods=1).mean().plot(figsize=(20,8),alpha = 1)


# In[22]:


df2=df1.reset_index()['Open']
df2


# In[23]:


plt.plot(df2)


# ## LSTM are sensitive to the scale of the data. so we apply MinMax scaler

# In[24]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df2=scaler.fit_transform(np.array(df2).reshape(-1,1))
print(df2)


# ## splitting dataset into train and test split

# In[25]:


train_size=int(len(df2)*0.75)
test_size=len(df2)-train_size
train_data,test_data=df2[0:train_size,:],df2[train_size:len(df2),:1]


# In[26]:


train_size,test_size


# In[27]:


train_data,test_data


# ## convert an array of values into a dataset matrix

# In[28]:


def create_dataset(dataset, time_step=1):
    train_X, train_Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        train_X.append(a)
        train_Y.append(dataset[i + time_step, 0])
    return numpy.array(train_X), numpy.array(train_Y)


# ## reshape into X=t,t+1,t+2,t+3 and Y=t+4

# In[29]:


import numpy
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[30]:


print(X_train.shape), print(y_train.shape)


# ## reshape input to be [samples, time steps, features] which is required for LSTM

# In[31]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# ## Create the Stacked LSTM model

# In[32]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[33]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[34]:


model.summary()


# In[35]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[36]:


import tensorflow as tf


# ## Lets Do the prediction and check performance metrics

# In[37]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[38]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# ## Calculate RMSE performance metrics

# In[39]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# ## Test Data RMSE

# In[40]:


math.sqrt(mean_squared_error(ytest,test_predict))


# ## shift train predictions for plotting

# In[41]:


look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict


# ## shift test predictions for plotting

# In[42]:


testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict


# ## plot baseline and predictions

# In[43]:


pred  = scaler.inverse_transform(df2)
plt.plot(pred,color='blue')
plt.show()


# In[44]:


plt.plot(trainPredictPlot,color='red')
plt.show()
plt.plot(testPredictPlot,color='green')
plt.show()


# In[45]:


plt.plot(trainPredictPlot,color='red')
plt.plot(testPredictPlot,color='green')
plt.show()


# In[46]:


plt.plot(pred,color='blue')
plt.plot(trainPredictPlot,color='red')
plt.plot(testPredictPlot,color='green')
plt.show()


# In[47]:


len(test_data)


# In[48]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# ## Save the Model

# In[49]:


model.save("saved_model.h5")


# In[ ]:




