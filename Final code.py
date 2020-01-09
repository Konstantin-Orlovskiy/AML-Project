# MODEL TUNING (KONSTANTIN ORLOVSKIY) - START
# LOADING TRAIN AND TEST DATA

# Train data.
import pandas as pd
data_train = pd.read_csv("FS_RFElog10_train_output.csv")
data_train.shape

data_train.head()

# split values into inpits and outputs.
values_train = data_train.values
X_train = values_train[:,1:11]
y_train = values_train[:,0]

data_train.shape

# Test data.
data_test = pd.read_csv("FS_RFElog10_test_output.csv")

# split values into inpits and outputs.
values_test = data_test.values
X_test = values_test[:,1:11]
y_test = values_test[:,0]

data_test.shape

# RANDOM FOREST IS THE BEST PERFORMING ALGORYTHM

## RF with default hyperparameters

# Initiate a RF model using default hyperparameters.
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# Train model on train data.
rf.fit(X_train, y_train)

# Check model accuracy on the TEST set.
rf_score = rf.score(X_test, y_test)
print(rf_score)

# Build confusion matrix.
from sklearn.metrics import confusion_matrix
rf_cm = confusion_matrix(y_test, rf.predict(X_test))
print(rf_cm)

## RF hyperparameters tuning (Random Search)

# Define a grid of hyperparameters.
rf_params = { 'n_estimators': [1, 5, 10, 30, 50, 100, 200, 500], 
             'max_depth': [None, 1, 2, 4, 8, 20, 50, 100], 
             'min_samples_leaf': [1, 5, 10, 50, 100], 
             'max_features': [None, 'auto', 'log2']
            }

# Run random search.
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, n_iter=25, 
                               cv = 3, n_jobs=-1, random_state = 2019)

import time
start_time = time.time()
rf_random.fit(X_train, y_train)
finish_time = time.time()

# Summarize results
print("Best: %f using %s" % (rf_random.best_score_, rf_random.best_params_))
print("Execution time: " + str((finish_time - start_time)))

# Apply best values of hyperparameters to the model.
rf_tuned = rf_random.best_estimator_

# Train the tuned model on TRAIN set and check the accuracy
rf_tuned.fit(X_train, y_train)
rf_tuned_score = rf_tuned.score(X_test,y_test)
print(rf_tuned_score)

# Build confusion matrix.
rf_tuned_cm = confusion_matrix(y_test, rf_tuned.predict(X_test))
print(rf_tuned_cm)

## RF tuning Results

print("RF default hyperparameters test accuracy: ", rf_score,', parameters: ', '\n', rf.get_params())
print('Confusion matrix: ', '\n', rf_cm)
print()
print("RF tuned hyperparameters test accuracy: ", rf_tuned_score,', parameters: ', '\n', rf_tuned.get_params())
print('Confusion matrix: ', '\n', rf_tuned_cm)

# MODEL TUNING (KONSTANTIN ORLOVSKIY) - END