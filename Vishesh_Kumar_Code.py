#!/usr/bin/env python
# coding: utf-8

# # Description:
# 
# *   In this notebook, we are going to predict whether a person's income is above 50k or below 50k using various features like age, education, and occupation.
# *  The dataset contains the labels which we have to predict and the labels are discrete and binary. So the problem we have is a Supervised Classification type.
# 
# 
# 
# 

# ## Importing Libraries

# In[1]:


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


# ### Importing the dataset

# In[2]:


df=pd.read_csv('train.csv')


# In[3]:


df.head()


# # Step 1: Data Exploration and Analysis
# 
# ---
# 
# 

# In[4]:



df.shape


# Merge the train and train_class_lables files 
# 

# In[5]:


df1 = pd.read_csv('train_class_labels.csv')


# In[6]:



df1.head()


# In[7]:


merged = pd.concat([df,df1], axis='columns')

merged.head()


# In[8]:


dataset=merged.drop(merged.columns[0], axis=1)


# In[9]:


dataset.head()


# In[10]:


dataset = dataset.rename(columns={'income_>50K': 'income'})
dataset.head()


# In[11]:


dataset.shape


# In[12]:


dataset.info()


# In[13]:


# Statistical summary
dataset.describe().T


# The number of missing data points per column

# In[14]:


#show the null values 
dataset.isnull().sum()


# In[15]:



# Checking the counts of label categories
income = dataset['income'].value_counts(normalize=True)
round(income * 100, 2).astype('str') + ' %'


# In[16]:




income.plot.pie(autopct='%1.1f%%')


# ## Observations:
# 
# The dataset have any null values in workclass, occupation, native-country, which needs to be preprocessed.
# 
# The dataset is unbalanced, as the dependent feature 'income' contains 76.07% values have income less than 50k and 23.93% values have income more than 50k.

# # Step 2: Data Visualization

# ## 2.1: Univariate Analysis

# In[17]:


# Creating a barplot for 'Income'
income = dataset['income'].value_counts()

plt.style.use('ggplot')
plt.figure(figsize=(7, 5))
sns.barplot(x=income.index, y=income.values, palette='dark')
plt.title('Distribution of Income', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Income', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.show()


# In[18]:


# Creating a distribution plot for 'Age'
age = dataset['age'].value_counts()

plt.figure(figsize=(10, 5))
plt.style.use('seaborn-bright')
sns.distplot(dataset['age'], bins=20)
plt.title('Distribution of Age', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.show()


# In[19]:


print('precise median age for >50K income : ', dataset[dataset['income']==1]['age'].median(), 'years')


# In[20]:


print('precise median age for <=50K income : ',dataset[dataset['income']==0]['age'].median(), 'years')


# MEDIAN value for people earning <=50K is around 34yrs, presicely.
# 
# MEDIAN value for people earing >50K is around 43yrs.

# In[21]:


# Creating a barplot for 'Education'
edu = dataset['education'].value_counts()

plt.style.use('seaborn-deep')
plt.figure(figsize=(10, 5))
sns.barplot(x=edu.values, y=edu.index, palette='bright')
plt.title('Distribution of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Education', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()


# In[22]:


# Creating a barplot for 'Education-num'
def draw_countplot(countplot_x, countplot_hue, countplot_data, figsize_a=20, figsize_b=10, xticks_rotation=45):
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(111)
    ax.set_facecolor((0,0,0.10))
    plt.xticks(rotation = xticks_rotation)
    plt.rc('xtick',labelsize=8)
    sns.countplot(x = countplot_x, hue = countplot_hue, data = countplot_data)
    plt.legend(prop={'size': 30})
    plt.show()


# In[23]:


draw_countplot(countplot_x='educational-num', countplot_hue='income', countplot_data=dataset)


# In[24]:


education_var = df['education'].unique()
df[df['education'] == 'Doctorate']


# In[25]:


education_var = df['education'].unique()
for edu_var in education_var:
    print("For {}, the Education Number is {}"
          .format(edu_var, df[df['education'] == edu_var]['educational-num'].unique()))


# I see that Education Number and Education are just the same, so, Education of them column could be droped.

# In[26]:



dataset.drop(['educational-num'], axis = 1, inplace = True)


# In[27]:



df['education'].value_counts()


# In[28]:



#Creating a countplot for 'workclass'
def draw_countplot(countplot_x, countplot_hue, countplot_data, figsize_a=20, figsize_b=10, xticks_rotation=45):
    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(111)
    ax.set_facecolor((0,0,0.10))
    plt.xticks(rotation = xticks_rotation)
    plt.rc('xtick',labelsize=8)
    sns.countplot(x = countplot_x, hue = countplot_hue, data = countplot_data)
    plt.legend(prop={'size': 30})
    plt.show()


# In[29]:


draw_countplot(countplot_x='workclass', countplot_hue='income', countplot_data=dataset)


# In[30]:


df['workclass'].value_counts()


# The two values Without-pay and Never-worked are insignificant and I can drop them.

# In[31]:


#df['workclass'].replace(['Never-worked', 'Without-pay'],'Other', inplace = True)
#df['workclass'].value_counts()


# In[32]:


# Creating a pie chart for 'Marital status'
marital = dataset['marital-status'].value_counts()

plt.style.use('fast')
plt.figure(figsize=(10, 7))
plt.pie(marital.values, labels=marital.index, startangle=10, explode=(
    0, 0.20, 0, 0, 0, 0, 0), shadow=True, autopct='%1.1f%%')
plt.title('Marital distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.legend()
plt.legend(prop={'size': 7})
plt.axis('equal')
plt.show()


# In[33]:


#Creating a countplot for 'OCCUPATION'
draw_countplot(countplot_x='occupation', countplot_hue='income', countplot_data=dataset)


# In[34]:


get_ipython().system('pip install squarify')


# In[35]:


# Creating a Treemap for 'Race'
import squarify
race = dataset['race'].value_counts()

plt.style.use('default')
plt.figure(figsize=(7, 5))
squarify.plot(sizes=race.values, label=race.index, value=race.values)
plt.title('Race distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.show()


# In[36]:



# Creating a barplot for 'Hours per week'
hours = dataset['hours-per-week'].value_counts().head(10)

plt.style.use('bmh')
plt.figure(figsize=(15, 7))
sns.barplot(x=hours.index, y=hours.values, palette='colorblind')
plt.title('Distribution of Hours of work per week', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Hours of work', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()


# Above dataset shows that most of the people work around 40 hours per weeks

# In[37]:


#dataset.head()


# In[38]:


dataset.head()


# ## 2.2 Bivariate Analysis

# In[39]:


# Creating a countplot of income across age
plt.style.use('default')
plt.figure(figsize=(20, 7))
sns.countplot(x='age', hue='income', data=dataset)
plt.title('Distribution of Income across Age', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# As we can observe from the graph 37-47 age people have more income

# In[40]:


# probability of belonging to the group with the highest income
workclass_income = dataset.groupby('workclass')['income'].mean() # there is correlation as expected

plt.rcParams['axes.axisbelow'] = True # grid behind graphs bars
plt.figure(figsize=(18, 7))
plt.ylim(0,1) # values from 0 to 1 as there are probabilities
plt.bar(workclass_income.index.astype(str), workclass_income,
       color = 'green' , edgecolor='black' )
plt.ylabel('Probability', size=20)
plt.xlabel('Workclass', size=20)
plt.grid(axis='y')


# As we can see in the above diagram self-employment peeps have higher probability of getting salary >50k

# In[41]:


# probability of belonging to the group with the highest income
education_income = dataset.groupby('education')['income'].mean()

plt.figure(figsize=(20, 8))
plt.ylim(0,1)
plt.xticks(rotation=30) # rotate axis text
plt.bar(education_income.index.astype(str), education_income,
       color = 'purple', edgecolor='black' )
plt.title('Income across Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.ylabel('Probability of earning >50k', size=20)
plt.xlabel('Education', size=20)
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.grid(axis='y')


# From the above graph we can infer that doctorate and prof-school educated people have more probability of getting salary of >50k

# In[42]:


# probability of belonging to the group with the highest income
marital_income = dict(dataset.groupby('marital-status')['income'].mean())
label = list(marital_income.keys())
slices = list(marital_income.values())
color=['red','green','yellow','blue','pink','skyblue','purple']
plt.pie(slices,labels=label,colors=color,radius=2,autopct="%0.2f%%")
plt.show()


# In[43]:


# Creating a countplot of income across sex
plt.style.use('fivethirtyeight')
plt.figure(figsize=(7, 3))
sns.countplot(x='gender', hue='income', data=dataset)
plt.title('Distribution of income across sex', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Sex', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 10})
plt.savefig('bi3.png')
plt.show()


# In[44]:


#Group categorical features by the output variable
cat = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender']
plt.figure(figsize = (20,20))
plotnumber = 1
for i in cat:
    plt.subplot(3,3,plotnumber)
    plt.xlabel(i)
    sns.countplot(x = 'income', hue = i, data = dataset)
    plotnumber+=1
plt.show()


# In[45]:


#Plotting a distribution of numerical values
numerical = ['age', 'fnlwgt',  'capital-gain', 'capital-loss', 'hours-per-week']
plt.figure(figsize = (20, 20))
plotnumber =1
for i in numerical:
    plt.subplot(3,3,plotnumber)
    plt.xlabel(i)
    sns.distplot(dataset[i])
    plotnumber+=1
plt.show()


# is there any missing values in income?

# Above pie chart shows marital statewise Probability of earning >50k
# Married people have more probability than the rest.

# In[46]:


missing_values_income = dataset['income'].isnull().sum()
print("Missing values in INCOME column ", missing_values_income)


# 
# 
# ## 2.3: Multivariate Analysis

# No missing values in INCOME column
# 
# Correlation matrix to identify their relation with income.
# 
# 

# In[47]:


plt.subplots(figsize=(20, 10))
sns.heatmap(dataset.corr(), vmax=.9, square=True, annot=True, fmt='.1f', center=0)
plt.show()

It’s clear that there is no high linear correlation between any of the continuous features and the target variable.
# # Step 3: Data Preprocessing

# ## 3.1: Fixing NaN values in the dataset

# In[48]:


#show the null values
dataset.isnull().sum()


# In[49]:


dataset.isnull().sum().sum()


# ### Filing missing value with the previous value of the column 

# In[50]:


df2= dataset.fillna(method = "pad")


# In[51]:


df2
df2.isnull().sum()


# In[52]:


df2


# In[53]:


columns_with_nan = ['workclass', 'occupation', 'native-country']


# In[54]:


for col in columns_with_nan:
    # Replacing with mode values as catogorical attributes are there
    dataset[col].fillna(dataset[col].mode()[0], inplace=True) 
    dataset.head()


# In[55]:


# Checking null values again
round((dataset.isnull().sum() / dataset.shape[0]) * 100, 2).astype(str) + ' %'


# ## 3.2 One Hot Encoding

# In[236]:


df = dataset


# In[237]:


target = 'income'
num_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country','education']


# In[238]:


print(len(cat_features))


# In[239]:


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


# In[240]:



encoder = OneHotEncoder()
encoder_cols = encoder.fit_transform(df[cat_features])


# In[241]:


# Add one-hot encoded columns to numerical features and target column
df = pd.concat([pd.concat([df[num_features], encoder_cols], axis=1), df[target]], axis=1)


# In[242]:


df.head()


# In[188]:


df.shape


# ## 3.3 Feature Selection

# In[189]:


features = df.columns.tolist()
features.remove(target)
X = df[features]


y = df[target]


# In[190]:


X.shape


# In[64]:


def ConfusionMatrix(classifier, X=X, y=y, confusionMatrix = True, plotConfusionMatrix = True):
        '''ConfusionMatrix function split data, fit data to model and give 
        a prediction for a given model and data. After that draw Confusion Matrix or 
        Plot Confusion Matrix to show the score'''
        
        #split dataset into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, 
                                                            stratify=y)

        # Fit the classifier to the data
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        #computing the confusion matrix with each row corresponding to the true class
        if(confusionMatrix):
            print(confusion_matrix(y_test, y_pred))

        #drawing Plot Confusion Matrix
        if(plotConfusionMatrix):
            plot_confusion_matrix(classifier, X_test, y_test)  
            plt.show() 


# In[65]:


def GridSearch(param, estimator, X=X, y=y):
    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    grid_rf = GridSearchCV(estimator, param, refit = True, verbose = 3, n_jobs=-1) 

    # fitting the model for grid search 
    grid_rf.fit(X_train, y_train) 

    # print best parameter after tuning 
    print(grid_rf.best_params_) 
    grid_rf_predictions = grid_rf.predict(X_test) 
    print('----------------------------------------')
    # print classification report 
    print(classification_report(y_test, grid_rf_predictions))


# In[66]:


from sklearn.ensemble import ExtraTreesClassifier
selector = ExtraTreesClassifier(random_state=42)


# In[67]:


selector.fit(X, y)


# In[68]:


feature_imp = selector.feature_importances_


# In[69]:


imp_feature = []


# In[70]:


for index, val in enumerate(feature_imp):
    print(index, round((val * 100), 2))
    imp_feature.append(round((val * 100), 2))


# In[71]:


X.info()


# In[72]:


imp_index = [i for i,j in enumerate(imp_feature) if j>1.0]
imp_index


# In[73]:


X_imp = X[X.columns[imp_index]]


# ## 3.4 Feature Scaling

# In[191]:


from sklearn.preprocessing import StandardScaler


# In[192]:


X.shape


# In[193]:


for col in X.columns:
    scaler = StandardScaler()
    X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))


# In[194]:


X.shape


# ## 3.5 Fixing imbalanced dataset using Oversampling

# In[195]:


round(y.value_counts(normalize=True) * 100, 2).astype('str') + ' %'


# In[196]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)


# In[197]:


ros.fit(X, y)


# In[198]:


X.shape


# In[199]:


X_resampled, Y_resampled = ros.fit_resample(X, y)


# In[200]:


round(Y_resampled.value_counts(normalize=True) * 100, 2).astype('str') + ' %'


# In[201]:


X_resampled.shape


# ## 3.6 Creating a train test split

# In[202]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_resampled, Y_resampled, test_size=0.2, random_state=42)


# In[203]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# In[204]:


Y_test


# # Step 4: Model Training

# ## 4.1 Logistic Regression

# In[94]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)


# In[95]:


log_reg.fit(X_train, Y_train)


# In[96]:


Y_pred_log_reg = log_reg.predict(X_test)
print(Y_pred_log_reg)


# ## 4.2 KNN Classifier

# In[97]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[98]:


knn.fit(X_train, Y_train)


# In[99]:


Y_pred_knn = knn.predict(X_test)


# ## 4.3 Support Vector Classifier

# In[229]:


from sklearn.svm import SVC
svc = SVC(random_state=42)


# In[230]:


svc.fit(X_train, Y_train)


# In[231]:


Y_pred_svc = svc.predict(X_test)


# ## 4.4 Decision Tree Classifier

# In[160]:


from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(random_state=42)


# In[161]:


dec_tree.fit(X_train, Y_train)


# In[162]:


Y_pred_dec_tree = dec_tree.predict(X_test)


# ## 4.5 Random Forest Classifier

# In[163]:


from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier(random_state=42)


# In[164]:


ran_for.fit(X_train, Y_train)


# In[165]:


X_train.shape


# In[166]:



Y_pred_ran_for = ran_for.predict(X_test)


# ## 4.6 Bernoulli Naive Bayes Classifier

# In[109]:


from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB()


# In[110]:


nb.fit(X_train, Y_train)


# In[111]:


Y_pred_nb = nb.predict(X_test)


# # Step 5: Model Evaluation
# 

# In[112]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[113]:


print('Logistic Regression:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_log_reg) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_log_reg) * 100, 2))


# In[114]:


print('KNN Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_knn) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_knn) * 100, 2))


# In[232]:


print('Support Vector Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_svc) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_svc) * 100, 2))


# In[115]:


print('Decision Tree Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_dec_tree) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_dec_tree) * 100, 2))


# In[116]:


print('Random Forest Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_ran_for) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_ran_for) * 100, 2))


# In[117]:


print('Bernoulli Naive Bayes Classifier:')
print('Accuracy score:', round(accuracy_score(Y_test, Y_pred_nb) * 100, 2))
print('F1 score:', round(f1_score(Y_test, Y_pred_nb) * 100, 2))


# # Step 6: Hyper parameter Tuning

# ## DECISION TREE
# 

# In[233]:



# Define grid of hyperparameters to search over
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [2, 3, 4, 5],
              'min_samples_leaf': [1, 2, 3, 4, 5]}


# Fit grid search object to data
GridSearch(param = param_grid, estimator = RandomForestClassifier())

# Print best hyperparameters and corresponding accuracy score
#print("Best parameters: ", grid.best_params_)
#print("Best accuracy score: ", grid.best_score_)


# ## RANDOM FOREST ALGORITHM

# In[259]:


param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [3, 4, 5, 6, 20,None],
              'min_samples_split': [2, 3, 4, 5],
              'min_samples_leaf': [1, 2, 3, 4, 5],
              'max_features': ['auto', 'sqrt', 'log2']}

GridSearch(param = param_grid, estimator = RandomForestClassifier())


# In[244]:


param = {'max_depth': [2, 10, 20],
         'n_estimators': [100, 500],
         'max_features': [10 , 20]}

GridSearch(param = param, estimator = RandomForestClassifier())


# In[245]:


param = {'max_depth': [20, 50],
         'max_features': [10, 30],
         'min_samples_split': [10, 30]}

GridSearch(param=param, estimator=RandomForestClassifier())


# In[246]:


#Random Forest model 
rand_forest = RandomForestClassifier(random_state=2020, n_jobs=-1, n_estimators=500, 
                                 max_depth=20, max_features=30, min_samples_split=30)


# In[247]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred_ran_for)
print(cm)


# ## K-NEAREST NEIGHBORS ALGORITHM

# In[250]:


#Standardization of data and KNN model 
pipe = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())
                ])

param = {'knn__n_neighbors': [10, 100],
         'knn__weights': ['uniform', 'distance'],
         'knn__p': [1, 2]}

GridSearch(param=param, estimator=pipe)


# In[251]:


#Standardization of data and KNN model 
pipe = Pipeline(steps=[
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier())
                ])

param = {'knn__n_neighbors': [10, 50],
         'knn__weights': ['uniform'],
         'knn__p': [1]}

GridSearch(param=param, estimator=pipe)


# In[252]:


#Standardization of data and KNN model 
knn_standarization = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=100, p=1, weights='uniform'))
                ])


# In[253]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred_knn)
print(cm)


# ## SUPPORT VECTOR MACHINE ALGORITHM

# In[254]:


# split into a training and testing set
pipe = Pipeline(steps=[
                ('scaler', StandardScaler()), 
                ('svc', SVC())
                ])

param = {'svc__C': [1, 10],
         'svc__kernel': ['linear', 'poly']}

GridSearch(param=param, estimator=pipe)


# In[256]:


# split into a training and testing set
pipe = Pipeline(steps=[
                ('scaler', StandardScaler()), 
                ('svc', SVC())
                ])

param = {'svc__kernel': ['linear', 'rbf'],
         'svc__gamma': ['scale', 'auto']
        }

GridSearch(param=param, estimator=pipe)


# In[257]:


#Standardization of data and KNN model 
svm_standarization = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('svm',  SVC(kernel='linear', C=10, gamma='scale', probability=True))
                ])


# In[258]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred_svc)
print(cm)


# # Conclusion

# 
# In this project, we build various models like logistic regression, knn classifier, support vector classifier, decision tree classifier, random forest classifier and Bernoulli Naive Bayes Classifier and hyper tune there parameters to find which models gives us the best results
# 
# After evaluating different models we find random forest classifier gives the highest accuracy score of 93 and f1 score of 93.28.
