#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:07:13 2019

@author: lisun
"""
""" 
Task 4: using exactly the same data sets (i.e. your training and testing sets) 
as in the last two tasks, implement Random Forest to rank all of your 
descriptive features based on their importance.
"""
 
import os
os.getcwd()
### change work directory to current 
os.chdir('/Users/lisun/GSU/2019 fall/Fundamental of Data Science/2019 Fall/HW3')

### read the training and testing data sets
import pandas as pd
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

### Rank all features based on their importance
from sklearn.ensemble import RandomForestClassifier
import numpy as np
feat_labels = X_train.columns
forest = RandomForestClassifier(n_estimators = 10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
rankTable = pd.DataFrame(columns=['Name','Value'])
for f in range(X_train.shape[1]):
    rankTable.loc[f]=[feat_labels[indices[f]], importances[indices[f]]]
    print('%2d) %-*s %f' %(f+1, 30, feat_labels[indices[f]], importances[indices[f]]))
    # *s specify <width> and .<precision> by variable
    # - flag causes the value to be left-justified in the specified field
rankTable.to_csv('rankTable.csv', index = False)    
    

# implement Random Forest 
resultsRF = pd.DataFrame(columns =['Count of Trees', 'Score for Training', 'Score for Testing'])   
for sizeOfForest in range(1,100):
    feat_labels = X_train.columns
    forest=RandomForestClassifier(n_estimators=sizeOfForest,random_state=0,n_jobs=-1)
    forest.fit(X_train,y_train)
    scoreTrain=forest.score(X_train, y_train)
    scoreTest=forest.score(X_test, y_test)
    resultsRF.loc[sizeOfForest]=[sizeOfForest,scoreTrain,scoreTest]
print(resultsRF.head(100))    
resultsRF.pop('Count of Trees')
ax9=resultsRF.plot(title = 'Random Forest').get_figure()
ax9.savefig('Random Forest.png')