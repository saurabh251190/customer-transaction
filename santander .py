#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing,CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


train = pd.read_csv('C:/Users/Saurabh Gautam/Desktop/Project2/train.csv')


# In[9]:


test= pd.read_csv('C:/Users/Saurabh Gautam/Desktop/Project2/test.csv')


# In[10]:


train['target'].value_counts()


# In[12]:


sns.countplot(train['target'])


# In[13]:


train.shape

train.describe()


# In[15]:


#missing values analysis
train.isnull().any().any()


# In[17]:


#variable distribution.
train.hist(figsize = (20,20), bins = 20)
plt.subplots_adjust(bottom=1.5, right=1.5, top=3)
plt.show()


# In[21]:


#Random forest classifier
#A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
label = train.target
features = [c for c in train.columns if c not in ['ID_code','target']]

X_train, X_test, y_train, y_test = train_test_split(train[features], label, test_size = 0.02, random_state = 7)
X_train1, y_train1 = X_train, y_train
X_test1, y_test1 = X_test, y_test

model1 = RandomForestClassifier(n_estimators = 50, random_state = 0).fit(X_train1, y_train1)
y_pred = model1.predict(X_test1)


# In[22]:


from sklearn.metrics import accuracy_score,roc_curve, auc
accuracy_score(y_test1, y_pred)


# In[23]:


feature_importances = pd.DataFrame(model1.feature_importances_, index = X_train.columns, columns = ['importance'])
feature_importances = feature_importances.sort_values('importance' , ascending = False)
#feature_importances.head()

colors = ['grey'] * 47 + ['green'] * 50
trace1 = go.Bar(x = feature_importances.importance[:97][::-1],
               y = [x.title()+"  " for x in feature_importances.index[:97][::-1]],
               name = 'feature importnace (relative)',
               marker = dict(color = colors, opacity=0.4), orientation = 'h')

data = [trace1]

layout = go.Layout(
    margin=dict(l=400), width = 1000, height = 1000,
    xaxis=dict(range=(0.0,0.015)),
    title='Feature Importance (Which Features are important to make predictions ?)',
    barmode='group',
    bargap=0.25
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:




