# -*- coding: utf-8 -*-
"""
Created on Wed July 22 20:16:47 2024

@author: bmahatwo
"""

import os
import pandas as pd
import numpy as np

## Get Input file
new_path = os.chdir(r'C:\Users\bmahatwo\OneDrive - Intel Corporation\Documents\Banashree Mahatwo - Project C\')
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

## Checking missing data
test_ids = test_set['PassengerId']

missing_data_train = train_set.isnull().sum()
print(missing_data_train)

missing_data_test = test_set.isnull().sum()
print(missing_data_test)

train_set['Embarked'].value_counts()

train_set['Fare'].value_counts()

## Fill up missing data
def lean(df):
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    df = df.copy()
    
    # Drop specified columns directly
    df.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1, inplace=True)

    # Fill missing 'Age' values with the median of 'Age'
    df['Age'] = df['Age'].fillna(df['Age'].median())

     # Fill missing 'Fare' values with the mean of 'Fare'
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    
    # Fill missing 'Embarked' values with 'S'
    df['Embarked'] = df['Embarked'].fillna('S')
    
    return df

train_set_new = lean(train_set)
test_set_new = lean(test_set)

missing_data = test_set_new.isnull().sum()
print(missing_data)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = ['Sex', 'Embarked']
for col in cols:
    train_set_new[col] = le.fit_transform(train_set_new[col])
    test_set_new[col] = le.fit_transform(test_set_new[col])

y = train_set_new['Survived']
X = train_set_new.drop(['Survived'], axis = 1)

print(y)

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

## Machine Learning Model - Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

## Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

submission_preds = classifier.predict(sc.transform(test_set_new))

df = pd.DataFrame({"PassengerId": test_ids.values,
                   "Survived": submission_preds})

df.to_csv('submission.csv',sep=',',encoding='utf-8',index = False)