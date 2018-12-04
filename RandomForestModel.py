#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:13:34 2018

@author: user
"""

#Script to carry out machine learning using Random Forest

import pandas as pd
import numpy as np

#Importing RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier

#Importing train-test split
from sklearn.model_selection import train_test_split

#Location of the data file on which classification algorithm needs to run on, please edit for to change to your local filepath
filepath_input='/User#Printing Colums of the AppBotDataset dataframe#Printing Colums of the AppBotDataset dataframes/user/Desktop/Big Data 2018/Capstone Project/Milestone4/ApplicationBotnetMergeFinal_33.csv'

#Storing CSV file in dataframe
AppBotDataset=pd.read_csv(filepath_input,header=0)

#Printing Colums of the AppBotDataset dataframe
print(AppBotDataset.columns)

#Selecting required features/attributes for classification
X=AppBotDataset[['Length', 'Answer RRs',
       'Time to live', 'Name Length']]

#Assigning target attribute
y=AppBotDataset[['Is botnet']]

#Train Test Split with 80-20 train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Creating RandomForest Classification model, with 100 estimators

clf = RandomForestClassifier(n_estimators=100)

#Fitting model
clf = clf.fit(X, y)

prediction=clf.predict(X_test)

from sklearn.metrics import classification_report

#Printing classification report which consists of various model performance measures.
report1 = classification_report(y_test, prediction)
print(report1)
