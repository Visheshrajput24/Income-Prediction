#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np
import pandas as pd
import os
from os import path

import sklearn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.neighbors import (KNeighborsClassifier, NeighborhoodComponentsAnalysis)
from sklearn.metrics import auc, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.impute import KNNImputer
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[69]:


df=pd.read_csv('train.csv')


# In[70]:


df_test = pd.read_csv('test.csv')


# In[71]:


labels = pd.read_csv('train_class_labels.csv')['income_>50K']


# In[72]:


del df['Unnamed: 0']
#del df['native-country']
del df["educational-num"]
del df_test["educational-num"]
#del df_test['native-country']


# In[73]:


df


# In[74]:


df_test


# In[75]:


num_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

cat_features = ['workclass', 'marital-status', 'occupation', 'relationship', "native-country", 'race', 'gender','education']


# In[76]:


for i in cat_features:
    print(i)
    print(df[i].unique().shape)
    print(df_test[i].unique().shape)


# In[77]:


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X = X)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns = new_columns, index = X.index)
        return d_out

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        return self.transform(X)

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f'{column}_{self.categories_[i][j]}')
                j += 1
        return new_columns


# In[78]:


encoder = OneHotEncoder()
encoder.fit(df[cat_features])


# In[89]:


X = pd.concat([encoder.transform(df[cat_features]), df[num_features]], axis=1)


# In[99]:


test = pd.concat([encoder.transform(df_test[cat_features]), df_test[num_features]] , axis=1)


# In[91]:


y = labels.values


# In[86]:


from sklearn.preprocessing import StandardScaler


# In[93]:


for col in X.columns:
    scaler = StandardScaler()
    X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))


# In[95]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)


# In[96]:


ros.fit(X, y)


# In[97]:


X_resampled, Y_resampled = ros.fit_resample(X, y)


# In[98]:


X_resampled.shape


# In[100]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_resampled, Y_resampled, test_size=0.2, random_state=42)


# In[101]:


from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(random_state=42)


# In[102]:


dec_tree.fit(X_train, Y_train)


# In[106]:


Y_pred_dec_tree = dec_tree.predict(X_test)


# In[107]:


print('Decision Tree Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_dec_tree) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_dec_tree) * 100, 2))


# In[108]:


from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier(random_state=42)
ran_for.fit(X_train, Y_train)


# In[109]:


Y_pred_ran_for = ran_for.predict(X_test)


# In[110]:


print('Random Forest Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_ran_for) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_ran_for) * 100, 2))


# In[111]:


predicted_labels = ran_for.predict(test)


# In[112]:


predicted_labels


# In[114]:


df_test


# In[118]:


with open("vishesh.txt", "w") as f:
    for i,j in enumerate(predicted_labels):
        f.write(f"{i},{j}\n")


# In[119]:


df_test.shape


# In[ ]:




