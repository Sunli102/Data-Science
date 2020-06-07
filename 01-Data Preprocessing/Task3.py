#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:46:22 2019

@author: lisun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import os

os.getcwd()
### change work directory to current 
os.chdir('/Users/lisun/Data Science')

### import the file 
df_Quan = pd.read_csv('Quantitative.csv')
df2 = df_Quan._get_numeric_data()

### Find outliers
outlier={}
for (name, series) in df2.iteritems():
    q1, q3 = np.percentile(df2[name], [25, 75])
    lwr = q1-1.5*(q3-q1)
    upr = q3+1.5*(q3-q1)
    outlier[name] = df2[name][(df2[name]<lwr) | (df2[name]>upr)]
    plt.boxplot(df2[name])
    plt.title('The Box Plot of '+df2[name].name)
    plt.savefig('BoxPlot of Original Data'+df2[name].name+'.png', dpi = 600)
    plt.show()
plt.close()
print(outlier)

### Implement clamp transformation
df2_clamp = pd.DataFrame()
for (name, series) in df2.iteritems():
    q1, q3 = np.percentile(df2[name], [25, 75])
    lwr = q1-1.5*(q3-q1)
    upr = q3+1.5*(q3-q1)
    df2_clamp[name] = np.clip(df2[name],lwr, upr)

### Normalize the data 
df2_clamp_Normalized = pd.DataFrame()
x = df2_clamp.values     # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df2_clamp_Normalized=pd.DataFrame(x_scaled, columns = df2_clamp.columns)   

### Generate box plots and SPLOMs on normalized data
# box plots
outlier_normalized = {}
for (name, series) in df2_clamp_Normalized.iteritems():
    q1, q3 = np.percentile(df2_clamp_Normalized[name], [25, 75])
    lwr = q1-1.5*(q3-q1)
    upr = q3+1.5*(q3-q1)
    outlier_normalized[name] = df2_clamp_Normalized[name][(df2_clamp_Normalized[name]<lwr) | (df2_clamp_Normalized[name]>upr)]
    plt.boxplot(df2_clamp_Normalized[name])
    plt.title('The Box Plot of Normalized '+df2_clamp_Normalized[name].name)
    plt.savefig('BoxPlot of Normalized Data'+df2_clamp_Normalized[name].name+'.png', dpi = 600)
    plt.show()
plt.close()
print(outlier_normalized)  

# SPLOMs 
sns.set(style='ticks')
Labels = df_Quan['Labels'].to_frame()
df2_clamp_Normalized_label=pd.merge(df2_clamp_Normalized,Labels,left_index=True, right_index=True)
ax = sns.pairplot(df2_clamp_Normalized_label, hue = 'Labels', markers =['+','o','*'],diag_kind = 'hist' )
ax.set_title =('SPLOM Of Normalized Data')
ax.savefig('SPLOM normalized.png', dpi = 600)


### save the results in a new file
QTransferred = pd.DataFrame()
for (name, series) in df2.iteritems():
    QTransferred[name]=df2[name]
    QTransferred[name+'_ClampedValues'] = df2_clamp[name]
    QTransferred[name+'_ClampedNormalizedValues'] = df2_clamp_Normalized[name]

QTransferred = pd.merge(QTransferred,Labels,left_index=True, right_index=True)
QTransferred.to_csv('QTransferred.csv')
