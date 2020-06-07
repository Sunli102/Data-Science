#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 02:47:57 2019

@author: lisun
"""

### Read the data from Others.csv file and generate all info 
### to create Summary Table in Data Quality Report for Categorical Features.
import pandas as pd
from  collections import Counter
import os

os.getcwd()
### change work directory to current 
os.chdir('/Users/lisun/GSU/2019 fall/Fundamental of Data Science/2019 Fall/HW2')

### import the file 
df_Others= pd.read_csv('Others.csv')
df_Others.head()
### create Summary Table in Data Quality Report 
for (name, series) in df_Others.iteritems():
    print('\n' + 'ANALYZED ATTRIBUTE NAME: ', name)
    print('-- COUNT: ', df_Others[name].size)
    print('-- % MISSING VALUES: ', df_Others[name].isnull().sum()/df_Others[name].size)
    print('-- CARDINALITY: ', df_Others[name].unique().size)
    print('-- MODE: ',Counter(df_Others[name]).most_common(1)[0][0]) #https://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
    print('-- MODE FREQ: ', Counter(df_Others[name]).most_common(1)[0][1])
    print('-- MODE %: ', Counter(df_Others[name]).most_common(1)[0][1]/df_Others[name].size)
    print('-- 2nd MODE: ', Counter(df_Others[name]).most_common(2)[1][0])
    print('-- 2nd MODE FREQ: ', Counter(df_Others[name]).most_common(2)[1][1])
    print('-- 2nd MODE %: ', Counter(df_Others[name]).most_common(2)[1][1]/df_Others[name].size)
   