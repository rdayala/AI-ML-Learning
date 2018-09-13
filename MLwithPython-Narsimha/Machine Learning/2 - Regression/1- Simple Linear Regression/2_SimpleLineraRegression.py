# -*- coding: utf-8 -*-
"""
Simple linear regression

One dependent variable (interval or ratio)
One independent variable (interval or ratio or dichotomous)

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data set

dataset = pd.read_csv('HR_DATA.csv')

X = dataset.iloc[:,:-1].values         # matrix of features / independent variables
y = dataset.iloc[:,1].values          # vector of dependent variables

# Splitting tha dataset into training set and test set
# training set is the one on which machine learning model learns
# test set is the one on which we check if ML learned correctly

from sklearn.cross_validation import train_test_split
X_train , X_test , y_train,y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)    # fit the linear model


# Predicting the test set results

y_pred = regressor.predict(X_test)

# visualizing the training set result

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing the test set result

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



























