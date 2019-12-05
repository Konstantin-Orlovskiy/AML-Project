import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#you need the pima_indians.csv to run

#label data
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age','label']
filename = 'pima-indians-diabetes.data.csv'
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']

#read data
data = read_csv(filename, names=col_names)
X = data[feature_cols]
y = data.label

# split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()

# fit model
logreg.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

#pass true and predicted values into dataframe
df = y_test.to_frame().reset_index()
df['predicted'] = y_pred_class
df['probabilty'] = y_pred_prob

#if you want to check data has read in correctly:
#df.head()

#to test function:
#model_evaluator(df)
