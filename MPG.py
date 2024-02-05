#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"C:\Users\Sharath\OneDrive\Desktop\MPG\Auto MPG Reg.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


#Convert horssepower into numeric
df.horsepower=pd.to_numeric(df.horsepower,errors="coerce")


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.horsepower=df.horsepower.fillna(df.horsepower.median())


# In[9]:


y=df.mpg
X=df.drop(["carname","mpg"],axis=1)


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


reg_model=LinearRegression().fit(X,y)


# In[12]:


reg_model.score(X,y)


# In[13]:


regpredict=reg_model.predict(X)


# In[14]:


from sklearn.metrics import mean_squared_error


# In[15]:


np.sqrt(mean_squared_error(y,regpredict))


# In[16]:


#For Deployment, model needs to be saved as.pkl(pickle)file or sav(joblib) library


# In[17]:


import joblib


# In[18]:


joblib.dump(reg_model,"reg.sav")







