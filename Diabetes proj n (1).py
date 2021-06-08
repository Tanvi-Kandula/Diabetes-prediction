#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
df=pd.read_csv("diabetes.csv")
df.head(10)


# In[21]:


print("Number of patients:"+str(len(df.index)))


# In[22]:


sns.countplot(x='Outcome',data=df)


# In[23]:


df.info()


# In[24]:


df.isnull().sum()


# In[25]:


df.head(5)


# In[26]:


df.drop("Pregnancies",axis=1,inplace=True)
df.head(5)


# In[27]:


X=df.drop("Outcome",axis=1)
y=df["Outcome"]


# In[28]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[11]:


from sklearn.linear_model import LogisticRegression


# In[12]:


log=LogisticRegression()


# In[13]:


log.fit(X_train,y_train)


# In[14]:


df.head(5)


# In[15]:


pred1=log.predict([[85,65,27,0,25.3,0.356,30]])
print(pred1)


# In[16]:


from sklearn.metrics import accuracy_score


# In[17]:


pred=log.predict(X_test)


# In[18]:


accuracy_score(y_test,pred)


# In[19]:


from sklearn.metrics import classification_report
classification_report(y_test,pred)


# In[ ]:





# In[ ]:





# In[ ]:




