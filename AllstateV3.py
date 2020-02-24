#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:02:28 2020

@author: amanprasad
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# load data
train_data = pd.read_csv('/Users/amanprasad/Documents/Courses_IIT_Fall_2019/ML/Assignment ML/HW3/train.csv',delimiter=',')



print("Train data dimensions: ", train_data.shape)


train_data.head()

print("Number of missing values",train_data.isnull().sum().sum())

train_data.describe()

contFeatureslist = []
for colName,x in train_data.iloc[1,:].iteritems():
    #print(x)
    if(not str(x).isalpha()):
        contFeatureslist.append(colName)
        

print(contFeatureslist)

contFeatureslist.remove("id")
contFeatureslist.remove("loss")

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize=(13,9))
sns.boxplot(train_data[contFeatureslist])

# Include  target variable also to find correlation between features and target feature as well
contFeatureslist.append("loss")

correlationMatrix = train_data[contFeatureslist].corr().abs()

plt.subplots(figsize=(13, 9))
sns.heatmap(correlationMatrix,annot=True)

# Mask unimportant features
sns.heatmap(correlationMatrix, mask=correlationMatrix< 1, cbar=False)
plt.show()
#correlationMatrix1 = train_data2[contFeatureslist].corr().abs()         

#drop correlated columns
# Select upper triangle of correlation matrix
upper = correlationMatrix.where(np.triu(np.ones(correlationMatrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]       

# Drop features 
train_data.drop(train_data[to_drop], axis=1, inplace=True)

train_data.columns




#Analysis of loss feature
plt.figure(figsize=(13,9))
sns.distplot(train_data["loss"])
sns.boxplot(train_data["loss"])

#Here, we can see loss is highly right skewed data. 
#This happened because there are many outliers in the data that we ca see from box plot. Lets apply log to see if we can get normal distribution

plt.figure(figsize=(13,9))
sns.distplot(np.log1p(train_data["loss"]))

#So we got normal distribution by applying logarithm on loss function
#We get normal distribution, so we can train model using target feature as log of loss. This way we don't have to remove outliers.
catCount = sum(str(x).isalpha() for x in train_data.iloc[1,:])
print("Number of categories: ",catCount)

#There are 116 categories with non alphanumeric values, most of the machine learning algorithms doesn't work with alpha numeric values. 
#So, lets convert it into numeric values

catFeatureslist = []
for colName,x in train_data.iloc[1,:].iteritems():
    if(str(x).isalpha()):
        catFeatureslist.append(colName)
        
#Unique categorical values per each category
print(train_data[catFeatureslist].apply(pd.Series.nunique)) 

#Convert categorical string values to numeric values  
from sklearn.preprocessing import LabelEncoder

for cf1 in catFeatureslist:
    le = LabelEncoder()
    le.fit(train_data[cf1].unique())
    train_data[cf1] = le.transform(train_data[cf1])   

train_data.head(5)

sum(train_data[catFeatureslist].apply(pd.Series.nunique) > 2)

#Analysis of categorical features with levels between 5-10

filterG5_10 = list((train_data[catFeatureslist].apply(pd.Series.nunique) > 5) & 
                (train_data[catFeatureslist].apply(pd.Series.nunique) < 10))

catFeaturesG5_10List = [i for (i, v) in zip(catFeatureslist, filterG5_10) if v]

len(catFeaturesG5_10List)

ncol = 2
nrow = 4
try:
    for rowIndex in range(nrow):
        f,axList = plt.subplots(nrows=1,ncols=ncol,sharey=True,figsize=(13, 9))
        features = catFeaturesG5_10List[rowIndex*ncol:ncol*(rowIndex+1)]
        
        for axIndex in range(len(axList)):
            sns.boxplot(x=features[axIndex], y="loss", data=train_data, ax=axList[axIndex])
                        
            # With original scale it is hard to visualize because of outliers
            axList[axIndex].set(yscale="log")
            axList[axIndex].set(xlabel=features[axIndex], ylabel='log loss')
except IndexError:
    print("")

#Correlation between categorical variables
filterG2 = list((train_data[catFeatureslist].apply(pd.Series.nunique) == 2))
catFeaturesG2List = [i for (i, v) in zip(catFeatureslist, filterG2) if v]
catFeaturesG2List.append("loss")

corrCatMatrix = train_data[catFeaturesG2List].corr().abs()

s = corrCatMatrix.unstack()
sortedSeries= s.sort_values(kind="quicksort",ascending=False)

print("Top 5 most correlated categorical feature pairs: \n")
print(sortedSeries[sortedSeries != 1.0][0:9])

#drop correlated columns
# Select upper triangle of correlation matrix
upper = corrCatMatrix.where(np.triu(np.ones(corrCatMatrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]       

# Drop features 
train_data.drop(train_data[to_drop], axis=1, inplace=True)

train_data.columns


