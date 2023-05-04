#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# In[2]:


df=pd.read_csv("insurance_claims.csv")
df


# In[3]:


df['insured_occupation']=pd.factorize(df['insured_occupation'])[0]
df['insured_education_level']=pd.factorize(df['insured_education_level'])[0]
df['auto_make']=pd.factorize(df['auto_make'])[0]
df['policy_state']=pd.factorize(df['policy_state'])[0]
df['auto_model']=pd.factorize(df['auto_model'])[0]
df['auto_year']=pd.factorize(df['auto_year'])[0]
df['insured_hobbies']=pd.factorize(df['insured_hobbies'])[0]
df['incident_type']=pd.factorize(df['incident_type'])[0]
df['collision_type']=pd.factorize(df['collision_type'])[0]
df['incident_severity']=pd.factorize(df['incident_severity'])[0]
df['authorities_contacted']=pd.factorize(df['authorities_contacted'])[0]
df['incident_state']=pd.factorize(df['incident_state'])[0]
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['month'] = pd.DatetimeIndex(df['incident_date']).month


# In[4]:


df_report=pd.get_dummies(df['police_report_available']).rename(columns = {'NO': 'no_report', 'YES': 'yes_report'})


# In[5]:


df=pd.concat([df,df_report],axis=1)


# In[6]:


sex=pd.get_dummies(df['insured_sex'],drop_first=True)
df=pd.concat([df,sex],axis=1)


# In[7]:


sex=pd.get_dummies(df['property_damage'],drop_first=True).rename(columns = {'NO': 'no_damage', 'YES': 'yes_damage'})
df=pd.concat([df,sex],axis=1)


# In[8]:


df=df.drop('policy_state',axis=1)
df=df.drop('policy_number',axis=1)
df=df.drop('policy_bind_date',axis=1)
df=df.drop('policy_csl',axis=1)
df=df.drop('_c39',axis=1)
df=df.drop('insured_zip',axis=1)
df=df.drop('insured_sex',axis=1)
df=df.drop('auto_model',axis=1)
df=df.drop('auto_year',axis=1)
df=df.drop('insured_relationship',axis=1)
df=df.drop('?',axis=1)
df=df.drop('police_report_available',axis=1)
df=df.drop('incident_date',axis=1)
df=df.drop('incident_city',axis=1)
df=df.drop('incident_location',axis=1)


# In[9]:


df.head(1)


# In[ ]:




