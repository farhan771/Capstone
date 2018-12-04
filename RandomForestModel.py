#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:13:34 2018

@author: user
"""

#Script to carry out machine learning using Random Forest

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
filepath_input='/Users/user/Desktop/Big Data 2018/Capstone Project/Milestone4/ApplicationBotnetMergeFinal_33.csv'
AppBotDataset=pd.read_csv(filepath_input,header=0)

print(AppBotDataset.columns)
X=AppBotDataset[['Length', 'Answer RRs',
       'Time to live', 'Name Length']]

y=AppBotDataset[['Is botnet']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X, y)

prediction=clf.predict(X_test)

from sklearn.metrics import classification_report

report = classification_report(y_test, prediction)

print(report)
