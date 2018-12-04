#Script to carry out machine learning using Catboost

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
filepath_input='/Users/user/Desktop/Big Data 2018/Capstone Project/Milestone4/ApplicationBotnetMergeFinal_33.csv'
AppBotDataset=pd.read_csv(filepath_input,header=0)

print(AppBotDataset.columns)
X=AppBotDataset[['Length', 'Answer RRs',
       'Time to live', 'Name Length']]

y=AppBotDataset[['Is botnet']]

#Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)


categorical_features_indices = np.where(X.dtypes != np.float)[0]


#model=CatBoostClassifier(iterations=50, depth=3, learning_rate=0.1, loss_function='Logloss')
from sklearn.metrics import accuracy_score
model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    random_seed=42,
    #logging_level='Silent'
    depth=3,
    iterations=50
)

model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test), plot=True);


'''
params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'eval_metric': 'Accuracy',
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': False
}
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
validate_pool = Pool(X_test, y_test, cat_features=categorical_features_indices)

model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=validate_pool)

best_model_params = params.copy()
best_model_params.update({
    'use_best_model': True
})
best_model = CatBoostClassifier(**best_model_params)
best_model.fit(train_pool, eval_set=validate_pool);

print('Simple model validation accuracy: {:.4}'.format(
    accuracy_score(y_test, model.predict(X_test))
))
'''
print('')

print('Best model validation accuracy: {:.4}'.format(
    accuracy_score(y_test, model.predict(X_test))
))

'''
cv_params = model.get_params()
cv_params.update({
    'loss_function': 'Logloss'
})
cv_data = cv(
    Pool(X, y, cat_features=categorical_features_indices),
    cv_params,
    plot=True
)

print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    np.max(cv_data['test-Accuracy-mean']),
    cv_data['test-Accuracy-std'][np.argmax(cv_data['test-Accuracy-mean'])],
    np.argmax(cv_data['test-Accuracy-mean'])

))
'''
prediction=model.predict(X_test)

from sklearn.metrics import classification_report

report = classification_report(y_test, prediction)

print(report)