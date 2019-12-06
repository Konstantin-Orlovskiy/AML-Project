# Import libraries
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Read in pre-processed training data
file = 'ScaledDataSet.csv'
data = pd.read_csv(file)

# Define our features and our target classifier
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

# First, remove all features with zero variance i.e. features with constant values. 
X = X.loc[:,X.apply(pd.Series.nunique) != 1]

# RFE with logistic regression
model = LogisticRegression()

# Try 10 features
rfe_log = RFE(model, 10)
fit_rfe_log = rfe_log.fit(X, Y)
# Create list with names of features
rfe_log_features = X.columns[fit_rfe_log.get_support()]

# Print names of selected features
for f in rfe_log_features:
    print(f)