#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:15:55 2019

@author: lisun
"""

""" 
Task 3: Now, using exactly the same data (i.e., training and testing data 
sets, which were generated and saved in Task 2, need to be loaded here), and 
evaluation methodology as in Task 2, investigate kNN classifiers. Perform 
experiments that involve at least two different similarity measures, different 
k values, and different neighbors‐weighting scenarios.
"""

import os
os.getcwd()
### change work directory to current 
os.chdir('/Users/lisun/GSU')

### read the training and testing data sets
import pandas as pd
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()


### KNN: EUCLIDEAN DISTANCE, MAJOIRITY VOTE
from sklearn.neighbors import KNeighborsClassifier
resultsKNNE = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnE = KNeighborsClassifier(n_neighbors = KNNDepth, p=2, metric = 'minkowski')
    knnE = knnE.fit(X_train, y_train)
    knnE.predict(X_test)
    scoreTrain = knnE.score(X_train, y_train)
    scoreTest = knnE.score(X_test, y_test)
    resultsKNNE.loc[KNNDepth]=[KNNDepth,scoreTrain, scoreTest]
print(resultsKNNE.head(21))
resultsKNNE.pop('LevelLimit')
ax3 = resultsKNNE.plot(title ='KNN_EUCLIDEAN WITH UNIFORM WEIGHTS').get_figure()
ax3.savefig('KNN_EUCLIDEAN.png')


### KNN: MANHATTAN DISTANCE
resultsKNNM = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnM = KNeighborsClassifier(n_neighbors = KNNDepth, p=1, metric = 'minkowski')
    knnM = knnM.fit(X_train, y_train)
    knnM.predict(X_test)
    scoreTrain = knnM.score(X_train, y_train)
    scoreTest = knnM.score(X_test, y_test)
    resultsKNNM.loc[KNNDepth]=[KNNDepth,scoreTrain, scoreTest]
print(resultsKNNM.head(21))
resultsKNNM.pop('LevelLimit')
ax4 = resultsKNNM.plot(title ='KNN_MANHATTAN WITH UNIFORM WEIGHTS').get_figure()
ax4.savefig('KNN_MANHATTAN.png')

### KNN: EUCLIDEAN DISTANCE WITH WEIGHTS
resultsKNNEW = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnEW = KNeighborsClassifier(n_neighbors = KNNDepth, p=2, metric = 'minkowski', weights = 'distance')
    knnEW  = knnEW .fit(X_train, y_train)
    knnEW .predict(X_test)
    scoreTrain = knnEW .score(X_train, y_train)
    scoreTest = knnEW .score(X_test, y_test)
    resultsKNNEW.loc[KNNDepth]=[KNNDepth,scoreTrain, scoreTest]
print(resultsKNNEW.head(21))
resultsKNNEW.pop('LevelLimit')
ax5 = resultsKNNEW.plot(title ='KNN_EUCLIDEAN WITH DISTANCE WEIGHTS').get_figure()
ax5.savefig('KNN_EUCLIDEAN_WEIGHTS.png')

### KNN: MANHATTAN DISTANCE WITH WEIGHTS
resultsKNNMW = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnMW = KNeighborsClassifier(n_neighbors = KNNDepth, p=1, metric = 'minkowski', weights = 'distance')
    knnMW = knnMW.fit(X_train, y_train)
    knnMW.predict(X_test)
    scoreTrain = knnMW.score(X_train, y_train)
    scoreTest = knnMW.score(X_test, y_test)
    resultsKNNMW .loc[KNNDepth]=[KNNDepth,scoreTrain,scoreTest]
print(resultsKNNMW .head(21))
resultsKNNMW .pop('LevelLimit')
ax6 = resultsKNNMW .plot(title ='KNN_MANHATTAN WITH DISTANCE WEIGHTS').get_figure()
ax6.savefig('KNN_MANHATTAN_WEIGHTS.png')



















