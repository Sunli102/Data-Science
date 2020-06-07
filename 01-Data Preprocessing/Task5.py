#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:52:25 2019

@author: lisun
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


os.getcwd()
os.chdir('/Users/lisun/GSU/2019 fall/Fundamental of Data Science/2019 Fall/HW2')

df4 = pd.read_csv('Quantitative.csv')
df_Quan = df4._get_numeric_data()

### Implement Equal-Frequency Binning on Quantitative.csv file
## Do Equal-Frequency Binning and smooth by mean
nn = 50

#QuantitativeBinned = pd.DataFrame(data = df_Quan)
outlier={}
QuantitativeBinned = pd.DataFrame(data = df_Quan)
for (name, series) in df_Quan.iteritems():
    df_Quan_sorted = pd.DataFrame()
    df_Quan_sorted[name+'_BIN'] = df_Quan[name].sort_values()
    for k in range(0,len(df_Quan_sorted[name+'_BIN']),nn):
        if k+nn <= len(df_Quan_sorted[name+'_BIN']):
            nn_mean = df_Quan_sorted[name+'_BIN'][k:k+nn].mean()
            df_Quan_sorted[name+'_BIN'][k:k+nn]=nn_mean
        else:
            bin_mean = df_Quan_sorted[name+'_BIN'][k:k+len(df_Quan_sorted[name+'_BIN'])].mean()
            df_Quan_sorted[name+'_BIN'][k:k+len(df_Quan_sorted[name+'_BIN'])]=bin_mean
#    print boxplot
    nbin = 20
    print('Attribute '+df_Quan_sorted[name+'_BIN'].name + ' with bins ', nbin)
    ax = df_Quan_sorted[name+'_BIN'].plot.hist(bins = nbin)
    ax.set_title =('Attribute: '+df_Quan_sorted[name+'_BIN'].name+'-histogram with '+str(nbin)+' bins')
    ax.set_xlabel(df_Quan_sorted[name+'_BIN'].name)
    ax.set_ylabel('Count')
    fig = ax.figure
    fig.set_size_inches(8,3)
    fig.tight_layout(pad = 1)
    fig.savefig('Equal Frequency histogram '+df_Quan_sorted[name+'_BIN'].name+'.png', dpi=600)
    plt.close(fig)
#    find outliers
    q1, q3 = np.percentile(df_Quan_sorted[name+'_BIN'], [25, 75])
    lwr = q1-1.5*(q3-q1)
    upr = q3+1.5*(q3-q1)
    outlier[name] = df_Quan_sorted[name+'_BIN'][(df_Quan_sorted[name+'_BIN']<lwr) | (df_Quan_sorted[name+'_BIN']>upr)]
    plt.boxplot(df_Quan_sorted[name+'_BIN'])
    plt.title('The Box Plot of '+df_Quan_sorted[name+'_BIN'].name)
    plt.savefig('BoxPlot of Binned Data'+df_Quan_sorted[name+'_BIN'].name+'.png', dpi = 600)
    plt.show()
#   print summary table of data quality for continuous features
    print('\n' + 'ANALYZED ATTRIBUTE NAME: ', name)
    print('-- SIZE: ', df_Quan_sorted[name+'_BIN'].size)
    print('-- MISSING VALUES: ', df_Quan_sorted[name+'_BIN'].isnull().sum())
    print('-- CARDINALITY: ', df_Quan_sorted[name+'_BIN'].unique().size)
    print('-- MEAN: ',df_Quan_sorted[name+'_BIN'].mean())
    print('-- SD: ', df_Quan_sorted[name+'_BIN'].std())
    print('-- Var: ', df_Quan_sorted[name+'_BIN'].var())
    print('-- MIN: ', df_Quan_sorted[name+'_BIN'].min())
    print('-- Q1: ', df_Quan_sorted[name+'_BIN'].quantile(0.25))
    print('-- MEDIAN: ', df_Quan_sorted[name+'_BIN'].quantile(0.5))
    print('-- Q3: ', df_Quan_sorted[name+'_BIN'].quantile(0.75))
    print('-- MAX: ',df_Quan_sorted[name+'_BIN'].max())
    
    
    binned = df_Quan_sorted[name+'_BIN'].to_frame().sort_index()
    QuantitativeBinned=pd.merge(QuantitativeBinned,binned,left_index=True, right_index = True)

## Generate new file, stored in the order 
Labels = df4['Labels'].to_frame()
QuantitativeBinned=pd.merge(QuantitativeBinned,Labels,left_index=True, right_index=True)
QuantitativeBinned.to_csv('QuantitativeBinned.csv')




### show scatter plot matrix (SPLOM); using hue with 'Labels'
binned_Quan=QuantitativeBinned.iloc[:,9:19]
sns.set(style='ticks')
ax = sns.pairplot(binned_Quan, hue = 'Labels', markers =['+','o','*'],diag_kind = 'hist' )
ax.set_title = ('SPLOM of Binned Data')
ax.savefig('SPLOM_Binned.png', dpi = 600)


### create covariance and correlaton tables
cov_table = binned_Quan.iloc[:,1:9].cov().round(decimals = 3)
print('cov_table\n', cov_table)
cov_table.to_csv('cov_table_binned.csv')
corr_table = binned_Quan.iloc[:,1:9].corr().round(decimals = 3)
print('corr_table\n', corr_table)
corr_table.to_csv('corr_table_binned.csv')

### plot heat maps for covariance and correlation tables
ax_cov= sns.heatmap(cov_table, linewidths=.5)
fig = ax_cov.get_figure()
fig.savefig('Heatmap of Covariance_binned', dpi = 600)

ax_corr = sns.heatmap(corr_table, linewidths=.5)
fig = ax_corr.get_figure()
fig.savefig('Heatmap of Correlation_binned', dpi= 600)
plt.close(fig)