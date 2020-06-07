#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:17:23 2019

@author: lisun
"""

"""
Task 2: Read the data from your wineNormalized.csv file
 to the Panda’s data frame and split your instances into training (2/3) 
 and testing (1/3) data sets (you will need to perform stratified 
 holdout sampling, as I want you to make sure you have an even‐out 
 number of class labels in each of these two sets). Save your training 
 and testing data sets as *.csv files, as you will need them to complete 
 the remaining tasks. In this task you will be working with Decision Trees,
 so perform experiments that involve trees with different numbers of l
 evels, and different homogeneity measures (e.g. Gini index, entropy).
"""
 

import os
os.getcwd()
### change work directory to current 
os.chdir('/Users/lisun/GSU/2019 fall/Fundamental of Data Science/2019 Fall/HW3')

### read the data from cancerNormalized.csv file to the Panda's data frame
import pandas as pd
df1 = pd.read_csv('cancerNormalized.csv')
print(df1.head(5))
print(df1.shape)
 

### split the instances into training(2/3) and test(1/3) data sets
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X,y = df1.iloc[:,1:31].values, df1['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 1/3, random_state = 5, stratify = y)
print(pd.DataFrame(data=X_train).shape)
print(pd.DataFrame(data=X_test).shape)

 
### save training and testing datasets as *.csv
pd.DataFrame(data = X_train, columns = df1.columns[1:31]).to_csv('X_train.csv', index = False)
pd.DataFrame(data = X_test,  columns = df1.columns[1:31]).to_csv('X_test.csv', index = False)
pd.DataFrame(data = y_train, columns = df1.columns[[0]]).to_csv('y_train.csv', index = False)
pd.DataFrame(data = y_test,  columns = df1.columns[[0]]).to_csv('y_test.csv', index = False)


### Decision Tree with Entropy
from sklearn.tree import DecisionTreeClassifier
resultsEntropy = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion = 'entropy', max_depth = treeDepth, random_state = 5)
    dct = dct.fit(X_train, y_train)
    dct.predict(X_test)
    scoreTrain = dct.score(X_train, y_train)
    scoreTest = dct.score(X_test, y_test)
    resultsEntropy.loc[treeDepth]=[treeDepth,scoreTrain, scoreTest]
print(resultsEntropy.head(11))
resultsEntropy.pop('LevelLimit')
ax1 = resultsEntropy.plot(title ='Desicion Tree_Entropy').get_figure()
ax1.savefig('DT_Entropy.png')


### Decision Tree with Gini
resultsGini = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion = 'gini', max_depth = treeDepth, random_state = 0)
    dct = dct.fit(X_train, y_train)
    dct.predict(X_test)
    scoreTrain = dct.score(X_train, y_train)
    scoreTest = dct.score(X_test, y_test)
    resultsGini.loc[treeDepth]=[treeDepth,scoreTrain, scoreTest]
print(resultsGini.head(11))
resultsGini.pop('LevelLimit')
ax2 = resultsGini.plot(title ='Desicion Tree_Gini').get_figure()
ax2.savefig('DT_Gini.png')



### graph visualization of decision trees
from sklearn.tree import export_graphviz
dctEntropy = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
dctEntropy = dctEntropy.fit(X_train, y_train)
export_graphviz(dctEntropy, out_file = 'treeEntropy.dot')
## directly render an existion DOT source file 
# https://graphviz.readthedocs.io/en/stable/manual.html
from graphviz import render
render('dot', 'png', 'treeEntropy.dot')
#predict class to test dataset
dctEntropy.predict(X_test)
#test score
score = dctEntropy.score(X_test, y_test)
print('Entropy testing score:',score)
score = dctEntropy.score(X_train, y_train)
print('Entropy training score:', score)


dctGini = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, random_state = 0)
dctGini = dctGini.fit(X_train, y_train)
export_graphviz(dctGini, out_file = 'treeGini.dot')
render('dot', 'png', 'treeGini.dot')
#predict class to test dataset
dctGini.predict(X_test)
#test score
score = dctGini.score(X_test, y_test)
print('Gini testing score:',score)
score = dctGini.score(X_train, y_train)
print('Gini training score:',score)


### make a pretty visualization
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image 
dot_data = StringIO()
export_graphviz(dctEntropy, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = df1.columns[1:31],class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('TreeEntropy1.png')
Image(graph.create_png())


dot_data = StringIO()
export_graphviz(dctGini, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = df1.columns[1:31],class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('TreeGini1.png')
Image(graph.create_png())












