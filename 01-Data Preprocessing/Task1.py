#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:14:30 2019

@author: lisun
"""

import pandas as pd
import numpy as np
import os

os.getcwd()
### change work directory to current 
os.chdir('/Users/lisun/GSU/')

### import the file and save it to a panda's data frame
df = pd.read_csv('dataPreP.csv')
df.dtypes

### identify attributes
Labels = df['Labels'].to_frame()
Quant1 = df.select_dtypes(include = [np.float64]) # not include target 'Labels'
Quantitative=pd.merge(Quant1,Labels, left_index = True, right_index=True)
Others = df.select_dtypes(include = ['object'])


### write to the files
path='Quantitative.csv'
Quantitative.to_csv(path,index = False)
path='Others.csv'
Others.to_csv(path, index = False)


### alternative ways to identify /seperate attributes
Others1 = df.loc[:, df.dtypes == 'object']
Quantitative1 = df.loc[:,df.dtypes == np.float64]
Quantitative2 = df.select_dtypes(exclude = ['object'])
