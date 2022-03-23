import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import ensemble, metrics
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import math
from numpy import random
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas import Series, DataFrame


# load data in

dataset = pd.read_csv('dvc_data/input/train.csv')
testset = pd.read_csv('dvc_data/input/test.csv')

# preprocessing of data

classcount1, classcount3, classcount2 = dataset['Y'].value_counts()

y1 = dataset[dataset['Y']==1] 
y2 = dataset[dataset['Y']==2] 
y3 = dataset[dataset['Y']==3] 

# oversample y2, y3

y2_oversample = y2.sample(classcount1, replace=True)
y3_oversample = y3.sample(classcount1, replace=True)

# append dataframe

dataset_oversample = pd.DataFrame(y1)
dataset_oversample = dataset_oversample.append(y2_oversample)
dataset_oversample = dataset_oversample.append(y3_oversample)

# shuffle data and assign variable

dataset_oversample = shuffle(dataset_oversample)
label = dataset_oversample.Y

# drop ID
dataset_oversample = dataset_oversample.drop('ID', axis=1)
testset = testset.drop('ID', axis=1)
dataset_oversample = dataset_oversample[dataset_oversample.columns[:46]]

kfold = KFold(n_splits = 20, shuffle = True)
predicted = []
expected = []

# # training model _ random forest

for train, test in kfold.split(dataset_oversample):
    x_train = dataset_oversample.iloc[train]
    y_train = label.iloc[train]
    x_test = dataset_oversample.iloc[test]
    y_test = label.iloc[test]
    forest = ensemble.RandomForestClassifier(n_estimators= 250, max_depth=8)
    forest1 = forest.fit(x_train,y_train)
    expected.extend(y_test)
    predicted.extend(forest.predict(x_test))

print("Macro-average: {0}".format(metrics.f1_score(expected,predicted,average='macro')))
print("Micro-average: {0}".format(metrics.f1_score(expected,predicted,average='micro')))
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))
accuracy = accuracy_score(expected, predicted)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print("Average = macro")
print('precision:',metrics.precision_score(expected, predicted,average='macro')) 
print('recall:',metrics.recall_score(expected, predicted,average='macro'))
print('F1-score:',metrics.f1_score(expected, predicted,labels=[1,2,3],average='macro'))

print("\n")
print("Average = micro")
print('precision:', metrics.precision_score(expected, predicted, average='micro')) 
print('recall:',metrics.recall_score(expected, predicted,average='micro'))
print('F1-score:',metrics.f1_score(expected, predicted,labels=[1,2,3],average='micro'))

print("\n")
print("Average = weighted")
print('precision:', metrics.precision_score(expected, predicted, average='weighted'))
print('recall:',metrics.recall_score(expected, predicted,average='weighted'))
print('F1-score:',metrics.f1_score(expected,predicted,labels=[1,2,3],average='weighted'))

# result = forest1.predict_proba(testset)
# print(result)

# c1 = []
# c2 = []
# c3 = []
# for i in range(300):
#     c1.append(result[i][0])
#     c2.append(result[i][1])
#     c3.append(result[i][2])


# # submission

# id = []
# for i in range(1,301):
#     id.append(i)

# submission = pd.DataFrame({
#     "ID": id,
#     "C1": c1,
#     "C2": c2,
#     "C3": c3,
# })
# submission.to_csv('dvc_data/input/result.csv', index=False)


