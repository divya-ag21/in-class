#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree  import DecisionTreeRegressor


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('canada_per_capita_income.csv')
df


# In[4]:


df.head(10)


# In[5]:


df.tail(10)


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


X = df.drop(columns = ['per capita income (US$)'])
Y = df['per capita income (US$)']


# In[11]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.15)


# In[12]:


X_train


# In[13]:


X_test.shape


# In[14]:


Y_train.shape


# In[15]:


Y_test.shape


# In[16]:


model_linear = LinearRegression()
model_tree = DecisionTreeRegressor()


# In[17]:


model_tree.fit(X_train,Y_train)


# In[18]:


model_linear.fit(X_train,Y_train)


# In[19]:


model_tree.predict([[1920]])


# In[20]:


model_tree.score(X_train,Y_train)


# In[21]:


model_linear.predict([[1920]])


# In[22]:


model_linear.score(X_train,Y_train)


# In[ ]:




