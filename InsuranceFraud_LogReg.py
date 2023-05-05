#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split


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
df['fraud_reported']=pd.factorize(df['fraud_reported'])[0]
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
df=df.drop('property_damage',axis=1)
df_end=df.copy()


# In[ ]:


pd.df_end.to_csv()


# In[9]:


df.head(10)


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


fraud=StandardScaler()


# In[12]:


unscaled=df_end.copy()
unscaled=unscaled.drop('fraud_reported',axis=1)
unscaled.head(10)


# In[13]:


target=df['fraud_reported']


# In[14]:


fraud.fit(unscaled)


# In[15]:


scaled=fraud.transform(unscaled)
scaled.shape


# In[17]:


train_test_split(scaled, target)


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(scaled, target,test_size = 0.2, random_state = 20)


# In[19]:


print (x_train.shape, y_train.shape)


# In[20]:


print (x_test.shape, y_test.shape)


# In[21]:


log_reg=LogisticRegression()


# In[22]:


log_reg.fit(x_train,y_train)


# In[23]:


log_reg.score(x_train,y_train)


# In[24]:


model_outputs = log_reg.predict(x_train)
model_outputs


# In[25]:


y_train


# In[26]:


model_outputs == y_train


# In[27]:


np.sum((model_outputs==y_train))


# In[28]:


model_outputs.shape[0]


# In[29]:


np.sum((model_outputs==y_train)) / model_outputs.shape[0]


# In[30]:


log_reg.intercept_


# In[31]:


log_reg.coef_


# In[32]:


unscaled.columns.values


# In[33]:


feature_name = unscaled.columns.values


# In[34]:


summary_table = pd.DataFrame (columns=['Feature name'], data = feature_name)


# In[36]:


summary_table['Coefficient'] = np.transpose(log_reg.coef_)


# In[37]:


summary_table


# In[38]:


summary_table.index = summary_table.index + 1


# In[39]:


summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
summary_table.sort_values('Odds_ratio', ascending=False)


# In[ ]:




