#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:13:07 2019

@author: lisun
"""
"""
Task 1: For the wine data set provided on our class website (wineData.csv): 
(1) open the file and read it to a Panda’s data frame, 
(2) identify Class attribute and perform class mapping, 
(3) normalize all remaining attributes to [0, 3] range using Min‐Max normalization, 
(4) save the entire data set as wineNormalized.csv file.
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

os.getcwd()
### change work directory to current 
os.chdir('/Users/lisun/GSU/2019 fall/Fundamental of Data Science/2019 Fall/HW3')


### import the file and save it to a panda's data frame
df = pd.read_csv('cancerData.csv')
print(df.head(10))
print(df.shape)

### identify attributes
print(df.dtypes)

### data mapping
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['Class']))} 
print(class_mapping)
ax = df['Class'].value_counts().plot(kind='bar', title ='Bar Chart of Class').get_figure()
print(df['Class'].value_counts())
df['Class']=df['Class'].map(class_mapping)
ax.savefig('barChart_classes.png')


###  Normalize to [0,3] range
df_Normalized = pd.DataFrame()
x = df.drop('Class', axis = 1).values # return a numpy array
x_column = df.drop('Class', axis = 1).columns  # hold on to columns names
## alternative way
#x = df.iloc[:,1:31].values        # returns s numpy array
#x_column = df.iloc[:,1:31].columns  # to hold on to columns names
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,3), copy = True) #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
x_scaled = min_max_scaler.fit_transform(x) 
df_temp = pd.DataFrame(x_scaled, columns = x_column)
df_Normalized = pd.concat([df['Class'], df_temp], axis = 1)
print(df_Normalized.head(5))
print(df_Normalized.tail(5))
print(df_Normalized.dtypes)
print(df_Normalized.shape)

### save the results in a new file
df_Normalized.to_csv('cancerNormalized.csv', index = False)


### plot pairplot
import seaborn as sns
sns.set(style = 'ticks')
ax = sns.pairplot(df_Normalized, hue='Class', diag_kind = 'hist')
#ax.set_title =('SPLOM Of Normalized Data')
ax.savefig('SPLOM cancerData_normalized.png')





