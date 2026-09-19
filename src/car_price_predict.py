#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[66]:


import matplotlib.pylab as plt


# In[87]:


df=pd.read_csv(r"C:\Users\yash\Downloads\quikr_car.csv")


# In[88]:


df.head()


# In[89]:


df.describe()


# In[90]:


df.info()


# In[91]:


df['fuel_type'].unique()


# #quality
# -year should be only in yeras
# -price should be in int
# -kms should be in int and without the comma
# - remove the fuel_type nan values
# - -only keep the first three word in the comapny

# In[92]:


backup=df.copy()


# In[93]:


df=df[df['year'].str.isnumeric()]


# In[94]:


df['year']=df['year'].astype(int)


# In[95]:


df.info()


# In[ ]:





# In[96]:


df=df[df['Price']!= "Ask For Price"]


# In[97]:


df['Price']=df['Price'].str.replace(',','').astype(int)


# In[98]:


df['Price']


# In[99]:


df['kms_driven']


# In[100]:


df['kms_driven']=df['kms_driven'].str.split(' ').str.get(0).str.replace(',','')


# In[ ]:





# In[103]:


df=df[df['kms_driven'].str.isnumeric()]


# In[104]:


df['kms_driven']


# In[105]:


df['kms_driven']=df['kms_driven'].astype(int)


# In[112]:


df['fuel_type'].unique()


# In[113]:


df=df[~df['fuel_type'].isna()]


# In[119]:


df['name']=df['name'].str.split(' ').str.slice(0,3).str.join(' ')


# In[122]:


df.reset_index(drop=True)


# In[125]:


df=df[df['Price']<6000000].reset_index(drop=True)


# In[126]:


df


# In[127]:


df.to_csv('clean_car.csv')


# In[128]:


x=df.drop('Price',axis=1)


# In[130]:


y=df['Price']


# In[132]:


from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV


# In[133]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[134]:


from sklearn.linear_model import LinearRegression


# In[135]:


lr=LinearRegression()


# In[137]:


from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder


# In[138]:


from sklearn.compose import ColumnTransformer


# In[149]:


one = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['name', 'company', 'fuel_type'])
    ],
    remainder='passthrough'
)


# In[146]:


from sklearn.pipeline import Pipeline


# In[150]:


pipe = Pipeline(
    steps=[
        ('preprocessor', one),
        ('model', lr)
    ]
)


# In[151]:


pipe.fit(x_train,y_train)


# In[153]:


y_pred=pipe.predict(x_test)


# In[156]:


r2_score(y_pred,y_test)


# In[159]:


scores=[]

for i in range(100):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
    pipe.fit(x_train,y_train)
    y_pred1=pipe.predict(x_test)
    scores.append(r2_score(y_pred1,y_test))



# In[160]:


np.argmax(scores)


# In[161]:


scores[np.argmax(scores)]


# In[162]:


import pickle


# In[163]:


pickle.dump(pipe,open('car_price.pkl','wb'))


# In[ ]:




