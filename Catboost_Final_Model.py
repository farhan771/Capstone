#Script to carry out machine learning using Catboost

import pandas as pd
import numpy as np

#Importing Catboost Classifier from Catboost Library
from catboost import CatBoostClassifier

#Importing Train Test Split from Scikit-Learn
from sklearn.model_selection import train_test_split

#Location of the data file on which classification algorithm needs to run on, please edit for to change to your local filepath
filepath_input='/Users/user/Desktop/Big Data 2018/Capstone Project/Milestone4/ApplicationBotnetMergeFinal_33.csv'

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

#Checking for attributes that are non-number
categorical_features_indices = np.where(X.dtypes != np.float)[0]


from sklearn.metrics import accuracy_score

#Creating Catboost model
model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    random_seed=42,
    #logging_level='Silent'
    depth=3,
    iterations=50
)

#Fitting model to train-test split

model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test), plot=True);


#To print accuracy score
print('Best model validation accuracy: {:.4}'.format(
    accuracy_score(y_test, model.predict(X_test))
))

#Prediction
prediction=model.predict(X_test)


from sklearn.metrics import classification_report

#Printing classification report which consists of various model performance measures.
report = classification_report(y_test, prediction)
print(report)
