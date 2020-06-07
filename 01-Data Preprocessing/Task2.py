#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:46:17 2019

@author: lisun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

os.getcwd()
### change work directory to current 
os.chdir('/Users/lisun/GSU/')

### import the file 
df_Quan = pd.read_csv('Quantitative.csv')
df_Quan.head()
df1=df_Quan._get_numeric_data()
df1.head()
### create Summary Table in Data Quality Report 
for (name, series) in df1.iteritems():
    print('\n' + 'ANALYZED ATTRIBUTE NAME: ', name)
    print('-- SIZE: ', df1[name].size)
    print('-- % MISSING VALUES: ', df1[name].isnull().sum()/df1[name].size)
    print('-- CARDINALITY: ', df1[name].unique().size)
    print('-- MEAN: ',df1[name].mean())
    print('-- SD: ', df1[name].std())
    print('-- Var: ', df1[name].var())
    print('-- MIN: ', df1[name].min())
    print('-- Q1: ', df1[name].quantile(0.25))
    print('-- MEDIAN: ', df1[name].quantile(0.5))
    print('-- Q3: ', df1[name].quantile(0.75))
    print('-- MAX: ', df1[name].max())


### plot equal-width histograms 
for (name, series) in df1.iteritems():
    # using Freedman-Diaconis rule to find the optimal number of bins 
    # ℎ=2*IQR*n^(−1/3) So the number of bins is (max−min)/ℎ
    # https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
    h = 2*np.subtract(*np.percentile(df1[name], [75, 25]))/(df1[name].size)**(1/3)
    nbin = math.ceil((max(df1[name])-min(df1[name]))/h)
    # plot histogrem with bin number = nbin
    print('Attribute '+df1[name].name + ' with bins ', nbin)
    ax = df1[name].plot.hist(bins = nbin)
    ax.set_title =('Attribute: '+df1[name].name+'-histogram with '+str(nbin)+' bins')
    ax.set_xlabel(df1[name].name)
    ax.set_ylabel('Count')
    fig = ax.figure
    fig.set_size_inches(8,3)
    fig.tight_layout(pad = 1)   
    fig.savefig('Equal width histogram '+df1[name].name+'.png', dpi=600)
    plt.close(fig)
    
 
    
### generate horizontal violin plots
for (name, series) in df1.iteritems():
    ax = sns.violinplot(x=df1[name]);
    ax.set_title('Horizontal Violin Plot for: '+ df1[name].name)
    fig = ax.figure
    fig.set_size_inches(8,3)
    fig.tight_layout(pad = 1)
    fig.savefig('Horizontal Violin Plot '+df1[name].name+'.png', dpi=600)
    plt.close(fig)

### show scatter plot matrix (SPLOM); using hue with 'Labels'
sns.set(style='ticks')
ax = sns.pairplot(df_Quan, hue = 'Labels', markers =['+','o','*'],diag_kind = 'hist' )
ax.set_title = ('SPLOM of Original Data')
ax.savefig('SPLOM.png', dpi = 600)


### create covariance and correlaton tables
cov_table = df1.cov().round(decimals = 3)
print('cov_table\n', cov_table)
cov_table.to_csv('cov_table.csv')
corr_table = df1.corr().round(decimals = 3)
print('corr_table\n', corr_table)
corr_table.to_csv('corr_table.csv')

### plot heat maps for covariance and correlation tables
ax_cov= sns.heatmap(cov_table, linewidths=.5)
fig = ax_cov.get_figure()
fig.savefig('Heatmap of Covariance', dpi = 600)

ax_corr = sns.heatmap(corr_table, linewidths=.5)
fig = ax_corr.get_figure()
fig.savefig('Heatmap of Correlation', dpi= 600)
plt.close(fig)


