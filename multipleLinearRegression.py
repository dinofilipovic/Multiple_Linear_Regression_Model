# Multiple Linear Regression

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Setting up information about current folder
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# defining initial workbook with csv file
initial_workbook = os.path.join(THIS_FOLDER, "50_Startups.csv")

# importing data into pandas for manipulation
dataset = pd.read_csv(initial_workbook)

# selecting the data set from 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(X)
# print(y)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# spliting test into test and train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
# create object of LinearRegression class, regressor is an instance of LinearRegression class
# we don't need to enter any paremeters in LinearRegression class
regressor = LinearRegression()
# this will train multiple linear regression
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
# This will display variables only with 2 decimals
np.set_printoptions(precision=2)
# Concatinate two vectors into one (I get X and Y coordinates)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
