#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:01:59 2019

@author: lisun
"""

"""
Task 5: Now, it is time to have some free play!􏰀Come up with some ideas to 
improve the machine learning results achieved in the tasks above, and test 
them. Try to (1) generate/derive some new descriptive features, 
(2) take advantage of feature ranking(s), or (3) other types of classifiers 
(feel free to use even the ones we did not discuss in the class), or numerosity
 reduction methods (e.g. binning, clustering) to further improve the results. 
"""
import os
os.getcwd()
### change work directory to current 
os.chdir('/Users/lisun/Data Science')

import pandas as pd
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()
rankTable = pd.read_csv('rankTable.csv')  # import Importance Rank Table

### improvement 1: drop the independent variables that are not important
cutoff = 0.005
columns=rankTable[rankTable['Value']<cutoff]['Name'].tolist()
newX_train = pd.DataFrame(X_train).drop(columns = columns)
newX_test = pd.DataFrame(X_test).drop(columns = columns)
newy_train = y_train
newy_test = y_test


## Decision Tree with Entropy
from sklearn.tree import DecisionTreeClassifier
resultsEntropy = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion = 'entropy', max_depth = treeDepth, random_state = 5)
    dct = dct.fit(newX_train, newy_train)
    dct.predict(newX_test)
    scoreTrain = dct.score(newX_train, newy_train)
    scoreTest = dct.score(newX_test, newy_test)
    resultsEntropy.loc[treeDepth]=[treeDepth,scoreTrain, scoreTest]

print(resultsEntropy.head(11))
resultsEntropy.pop('LevelLimit')
ax1 = resultsEntropy.plot(title ='Desicion Tree_Entropy').get_figure()
#ax1.savefig('DT_Entropy.pdf')


## Decision Tree with Gini
resultsGini = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion = 'gini', max_depth = treeDepth, random_state = 0)
    dct = dct.fit(newX_train, newy_train)
    dct.predict(newX_test)
    scoreTrain = dct.score(newX_train, newy_train)
    scoreTest = dct.score(newX_test, newy_test)
    resultsGini.loc[treeDepth]=[treeDepth,scoreTrain, scoreTest]

print(resultsGini.head(11))
resultsGini.pop('LevelLimit')
ax2 = resultsGini.plot(title ='Desicion Tree_Gini').get_figure()
#ax2.savefig('DT_Gini.pdf')

## KNN: EUCLIDEAN DISTANCE, MAJOIRITY VOTE
from sklearn.neighbors import KNeighborsClassifier
resultsKNNE = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnE = KNeighborsClassifier(n_neighbors = KNNDepth, p=2, metric = 'minkowski')
    knnE = knnE.fit(newX_train, newy_train)
    knnE.predict(newX_test)
    scoreTrain = knnE.score(newX_train, newy_train)
    scoreTest = knnE.score(newX_test, newy_test)
    resultsKNNE.loc[KNNDepth]=[KNNDepth,scoreTrain, scoreTest]
print(resultsKNNE.head(21))
resultsKNNE.pop('LevelLimit')
ax3 = resultsKNNE.plot(title ='KNN_EUCLIDEAN DISTANCE').get_figure()
#ax3.savefig('KNN_EUCLIDEAN.pdf')


## KNN: MANHATTAN DISTANCE
resultsKNNM = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnM = KNeighborsClassifier(n_neighbors = KNNDepth, p=1, metric = 'minkowski')
    knnM = knnM.fit(newX_train, newy_train)
    knnM.predict(newX_test)
    scoreTrain = knnM.score(newX_train, newy_train)
    scoreTest = knnM.score(newX_test, newy_test)
    resultsKNNM.loc[KNNDepth]=[KNNDepth,scoreTrain, scoreTest]
print(resultsKNNM.head(21))
resultsKNNM.pop('LevelLimit')
ax4 = resultsKNNM.plot(title ='KNN_MANHATTAN DISTANCE').get_figure()
#ax4.savefig('KNN_MANHATTAN.pdf')

## KNN: EUCLIDEAN DISTANCE WITH WEIGHTS
resultsKNNEW = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnEW = KNeighborsClassifier(n_neighbors = KNNDepth, p=2, metric = 'minkowski', weights = 'distance')
    knnEW  = knnEW .fit(newX_train, newy_train)
    knnEW .predict(newX_test)
    scoreTrain = knnEW .score(newX_train, newy_train)
    scoreTest = knnEW .score(newX_test, newy_test)
    resultsKNNEW.loc[KNNDepth]=[KNNDepth,scoreTrain, scoreTest]
print(resultsKNNEW.head(21))
resultsKNNEW.pop('LevelLimit')
ax5 = resultsKNNEW.plot(title ='KNN_EUCLIDEAN WITH WEIGHTS').get_figure()
#ax5.savefig('KNN_EUCLIDEAN_WEIGHTS.pdf')

## KNN: MANHATTAN DISTANCE WITH WEIGHTS
resultsKNNMW = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnMW = KNeighborsClassifier(n_neighbors = KNNDepth, p=1, metric = 'minkowski', weights = 'distance')
    knnMW = knnMW.fit(newX_train, newy_train)
    knnMW.predict(newX_test)
    scoreTrain = knnMW.score(newX_train, newy_train)
    scoreTest = knnMW.score(newX_test, newy_test)
    resultsKNNMW .loc[KNNDepth]=[KNNDepth,scoreTrain,scoreTest]
print(resultsKNNMW .head(21))
resultsKNNMW .pop('LevelLimit')
ax6 = resultsKNNMW .plot(title ='KNN_MANHATTAN WITH WEIGHTS').get_figure()
#ax6.savefig('KNN_MANHATTAN_WEIGHTS.pdf')

## Random Forest 
from sklearn.ensemble import RandomForestClassifier
resultsRF = pd.DataFrame(columns =['Count of Trees', 'Score for Training', 'Score for Testing'])   
for sizeOfForest in range(1,100):
    feat_labels = newX_train.columns
    forest=RandomForestClassifier(n_estimators=sizeOfForest,random_state=0,n_jobs=-1)
    forest.fit(newX_train,newy_train)
    scoreTrain=forest.score(newX_train, newy_train)
    scoreTest=forest.score(newX_test, newy_test)
    resultsRF.loc[sizeOfForest]=[sizeOfForest,scoreTrain,scoreTest]
print(resultsRF.head(50))    
resultsRF.pop('Count of Trees')
ax9=resultsRF.plot(title = 'Random Forest').get_figure()
#ax9.savefig('Random Forest.pdf')


# Try to use ADABOOST
from sklearn.ensemble import AdaBoostClassifier
resultsAda = pd.DataFrame(columns =['Count of Trees', 'Score for Training', 'Score for Testing'])
indexR = 1
for countOfTrees in range(1, 102, 10):
    tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 1, random_state = 0)
    ada = AdaBoostClassifier(base_estimator=tree, n_estimators=countOfTrees, learning_rate = 0.5)
    ada.fit(newX_train, newy_train)
    scoreTrain = ada.score(newX_train, newy_train)
    scoreTest = ada.score(newX_test, newy_test)
    resultsAda.loc[indexR] = [sizeOfForest, scoreTrain, scoreTest]
    indexR = indexR +1
print(resultsAda.head(11))
resultsAda.pop('Count of Trees')
ax10 = resultsAda.plot(title = 'Random Forest_Entropy').get_figure()


### 
## create correlaton tables
cancerNormalized = pd.read_csv('cancerNormalized.csv')
cancerNew = cancerNormalized.drop(columns =columns)
corr_table = cancerNew.corr().round(decimals = 3)
print('corr_table\n', corr_table)
corr_table.to_csv('corr_table.csv')

## plot heat maps for covariance and correlation tables
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
mask = np.zeros_like(corr_table) # https://seaborn.pydata.org/generated/seaborn.heatmap.html
mask[np.triu_indices_from(mask)] = True
ax_corr = sns.heatmap(corr_table, linewidths=.3, mask=mask, annot=True)
fig = ax_corr.get_figure()
fig.savefig('Heatmap of Correlation1', dpi= 600)
#plt.close(fig)

## deal with multicollinearity
newX_train['shape2'] = newX_train['radius2']+newX_train['perimeter2']+newX_train['area2']
newX_train['shape3'] = newX_train['radius3']+newX_train['perimeter3']+newX_train['area3']
newX_train = newX_train.drop(columns = ['radius1','perimeter1','area1'])
newX_train = newX_train.drop(columns = ['radius2','perimeter2','area2'])
newX_train = newX_train.drop(columns = ['radius3','perimeter3','area3'])
newX_train['texture13'] = newX_train['texture1']+newX_train['texture3']
newX_train = newX_train.drop(columns = ['texture1','texture3'])
newX_train['concavity123'] = newX_train['concavity1']+newX_train['concavity2']+newX_train['concavity3']
newX_train = newX_train.drop(columns = ['concavity1','concavity2','concavity3'])
newX_train['concave points13'] = newX_train['concave points1']+newX_train['concave points3']
newX_train = newX_train.drop(columns = ['concave points1','concave points3'])
newX_train['compactness13'] = newX_train['compactness1']+newX_train['compactness3']
newX_train = newX_train.drop(columns = ['compactness1','compactness3'])


newX_test['shape2'] = newX_test['radius2']+newX_test['perimeter2']+newX_test['area2']
newX_test['shape3'] = newX_test['radius3']+newX_test['perimeter3']+newX_test['area3']
newX_test = newX_test.drop(columns = ['radius1','perimeter1','area1'])
newX_test = newX_test.drop(columns = ['radius2','perimeter2','area2'])
newX_test = newX_test.drop(columns = ['radius3','perimeter3','area3'])
newX_test['texture13'] = newX_test['texture1']+newX_test['texture3']
newX_test = newX_test.drop(columns = ['texture1','texture3'])
newX_test['concavity123'] = newX_test['concavity1']+newX_test['concavity2']+newX_test['concavity3']
newX_test = newX_test.drop(columns = ['concavity1','concavity2','concavity3'])
newX_test['concave points13'] = newX_test['concave points1']+newX_test['concave points3']
newX_test = newX_test.drop(columns = ['concave points1','concave points3'])
newX_test['compactness13'] = newX_test['compactness1']+newX_test['compactness3']
newX_test = newX_test.drop(columns = ['compactness1','compactness3'])

newX_test.shape

## Decision Tree with Entropy
from sklearn.tree import DecisionTreeClassifier
resultsEntropy = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion = 'entropy', max_depth = treeDepth, random_state = 5)
    dct = dct.fit(newX_train, newy_train)
    dct.predict(newX_test)
    scoreTrain = dct.score(newX_train, newy_train)
    scoreTest = dct.score(newX_test, newy_test)
    resultsEntropy.loc[treeDepth]=[treeDepth,scoreTrain, scoreTest]
print(resultsEntropy.head(11))
resultsEntropy.pop('LevelLimit')
ax1 = resultsEntropy.plot(title ='Desicion Tree_Entropy').get_figure()
#ax1.savefig('DT_Entropy.pdf')

## Decision Tree with Gini
resultsGini = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion = 'gini', max_depth = treeDepth, random_state = 0)
    dct = dct.fit(newX_train, newy_train)
    dct.predict(newX_test)
    scoreTrain = dct.score(newX_train, newy_train)
    scoreTest = dct.score(newX_test, newy_test)
    resultsGini.loc[treeDepth]=[treeDepth,scoreTrain, scoreTest]
print(resultsGini.head(11))
resultsGini.pop('LevelLimit')
ax2 = resultsGini.plot(title ='Desicion Tree_Gini').get_figure()
#ax2.savefig('DT_Gini.pdf')

## KNN: EUCLIDEAN DISTANCE, MAJOIRITY VOTE
from sklearn.neighbors import KNeighborsClassifier
resultsKNNE = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnE = KNeighborsClassifier(n_neighbors = KNNDepth, p=2, metric = 'minkowski')
    knnE = knnE.fit(newX_train, newy_train)
    knnE.predict(newX_test)
    scoreTrain = knnE.score(newX_train, newy_train)
    scoreTest = knnE.score(newX_test, newy_test)
    resultsKNNE.loc[KNNDepth]=[KNNDepth,scoreTrain, scoreTest]
print(resultsKNNE.head(21))
resultsKNNE.pop('LevelLimit')
ax3 = resultsKNNE.plot(title ='KNN_EUCLIDEAN DISTANCE').get_figure()
#ax3.savefig('KNN_EUCLIDEAN.pdf')


## KNN: MANHATTAN DISTANCE
resultsKNNM = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnM = KNeighborsClassifier(n_neighbors = KNNDepth, p=1, metric = 'minkowski')
    knnM = knnM.fit(newX_train, newy_train)
    knnM.predict(newX_test)
    scoreTrain = knnM.score(newX_train, newy_train)
    scoreTest = knnM.score(newX_test, newy_test)
    resultsKNNM.loc[KNNDepth]=[KNNDepth,scoreTrain, scoreTest]
print(resultsKNNM.head(21))
resultsKNNM.pop('LevelLimit')
ax4 = resultsKNNM.plot(title ='KNN_MANHATTAN DISTANCE').get_figure()
#ax4.savefig('KNN_MANHATTAN.pdf')

## KNN: EUCLIDEAN DISTANCE WITH WEIGHTS
resultsKNNEW = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnEW = KNeighborsClassifier(n_neighbors = KNNDepth, p=2, metric = 'minkowski', weights = 'distance')
    knnEW  = knnEW .fit(newX_train, newy_train)
    knnEW .predict(newX_test)
    scoreTrain = knnEW .score(newX_train, newy_train)
    scoreTest = knnEW .score(newX_test, newy_test)
    resultsKNNEW.loc[KNNDepth]=[KNNDepth,scoreTrain, scoreTest]
print(resultsKNNEW.head(21))
resultsKNNEW.pop('LevelLimit')
ax5 = resultsKNNEW.plot(title ='KNN_EUCLIDEAN WITH WEIGHTS').get_figure()
#ax5.savefig('KNN_EUCLIDEAN_WEIGHTS.pdf')

## KNN: MANHATTAN DISTANCE WITH WEIGHTS
resultsKNNMW = pd.DataFrame(columns=['LevelLimit','Score for Training','Score for Testing'])
for KNNDepth in range (1,21):
    knnMW = KNeighborsClassifier(n_neighbors = KNNDepth, p=1, metric = 'minkowski', weights = 'distance')
    knnMW = knnMW.fit(newX_train, newy_train)
    knnMW.predict(newX_test)
    scoreTrain = knnMW.score(newX_train, newy_train)
    scoreTest = knnMW.score(newX_test, newy_test)
    resultsKNNMW .loc[KNNDepth]=[KNNDepth,scoreTrain,scoreTest]
print(resultsKNNMW .head(21))
resultsKNNMW .pop('LevelLimit')
ax6 = resultsKNNMW .plot(title ='KNN_MANHATTAN WITH WEIGHTS').get_figure()
#ax6.savefig('KNN_MANHATTAN_WEIGHTS.pdf')

## Random Forest 
from sklearn.ensemble import RandomForestClassifier
resultsRF = pd.DataFrame(columns =['Count of Trees', 'Score for Training', 'Score for Testing'])   
for sizeOfForest in range(1,50):
    feat_labels = newX_train.columns
    forest=RandomForestClassifier(n_estimators=sizeOfForest,random_state=0,n_jobs=-1)
    forest.fit(newX_train,newy_train)
    scoreTrain=forest.score(newX_train, newy_train)
    scoreTest=forest.score(newX_test, newy_test)
    resultsRF.loc[sizeOfForest]=[sizeOfForest,scoreTrain,scoreTest]
print(resultsRF.head(50))    
resultsRF.pop('Count of Trees')
ax11=resultsRF.plot(title = 'Random Forest').get_figure()

## Try to use ADABOOST
from sklearn.ensemble import AdaBoostClassifier
resultsAda = pd.DataFrame(columns =['Count of Trees', 'Score for Training', 'Score for Testing'])
indexR = 1
for countOfTrees in range(1, 102, 10):
    tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 1, random_state = 0)
    ada = AdaBoostClassifier(base_estimator=tree, n_estimators=countOfTrees, learning_rate = 0.1)
    ada.fit(newX_train, newy_train)
    scoreTrain = ada.score(newX_train, newy_train)
    scoreTest = ada.score(newX_test, newy_test)
    resultsAda.loc[indexR] = [sizeOfForest, scoreTrain, scoreTest]
    indexR = indexR +1
print(resultsAda.head(11))
resultsAda.pop('Count of Trees')
ax12 = resultsAda.plot(title = 'Random Forest_Entropy').get_figure()


##########################
### logisitic Regression
import pandas as pd
cancerNormalized = pd.read_csv('cancerNormalized.csv')
y = cancerNormalized.pop('Class')
X = cancerNormalized

## implemet RFE for varaible selection
"""
Recursive Feature Eliminaton: select features by recursively considering 
smaller and smaller sets of features
"""
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='lbfgs')

rfe = RFE(logreg, n_features_to_select = 20)
rfe = rfe.fit(X, y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

cols = X.columns[rfe.support_.tolist()]

newy = y
newX = X[cols]



### handle multicollinearity
## plot heat maps for covariance and correlation tables
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
corr_table = newX.corr().round(decimals = 3)
print('corr_table\n', corr_table)
corr_table.to_csv('corr_table.csv')
plt.figure(figsize=(15,8))
mask = np.zeros_like(corr_table) # https://seaborn.pydata.org/generated/seaborn.heatmap.html
mask[np.triu_indices_from(mask)] = True
ax_corr = sns.heatmap(corr_table, linewidths=.3, mask=mask, annot=True)
fig = ax_corr.get_figure()

## do feature aggregation 
newX['perimeter123'] = newX['perimeter1']+newX['perimeter2']+newX['perimeter3']
newX = newX.drop(columns = ['radius1','perimeter1','area1'])
newX = newX.drop(columns = ['radius2','perimeter2','area2'])
newX = newX.drop(columns = ['radius3','perimeter3','area3'])
newX['texture13'] = newX['texture1']+newX['texture3']
newX = newX.drop(columns = ['texture1','texture3'])
newX['concavity13'] = newX['concavity1']+newX['concavity3']
newX = newX.drop(columns = ['concavity1','concavity3'])
newX['concave points13'] = newX['concave points1']+newX['concave points3']
newX = newX.drop(columns = ['concave points1','concave points3'])
newX['smoothness13'] = newX['smoothness1']+newX['smoothness3']
newX = newX.drop(columns = ['smoothness1','smoothness3'])

newX.columns
newX.shape

## check new dataset with new features
corr_table = newX.corr().round(decimals = 3)
print('corr_table\n', corr_table)
corr_table.to_csv('corr_table.csv')
plt.figure(figsize=(15,8))
mask = np.zeros_like(corr_table) # https://seaborn.pydata.org/generated/seaborn.heatmap.html
mask[np.triu_indices_from(mask)] = True
ax_corr = sns.heatmap(corr_table, linewidths=.3, mask=mask, annot=True)
fig = ax_corr.get_figure()


## implementing the model selection and remove the variables with big p-value
import statsmodels.api as sm
logit_model = sm.Logit(newy, newX)
result = logit_model.fit(method = 'lbfgs')
print(result.summary2())

## logisitic Regression Model Fitting
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(newX,newy, test_size= 1/3, random_state = 5, stratify = y)

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)


## predict the test set results and calculate the accuracy
y_pred = logreg.predict(X_test)
trainScore = logreg.score(X_train, y_train)
testScore = logreg.score(X_test, y_test)
print('Accuracy of logistic regression classifier on train set: {:.4f}'.format(trainScore))
print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(testScore))
#Accuracy of logistic regression classifier on train set: 0.9716
#Accuracy of logistic regression classifier on test set: 0.9648


## confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

## Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

## ROC Receiver Operation Characteristic
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
