# -*- coding: utf-8 -*-
"""

Polynomial Linear` Regression

y = b0+ b1x1 + b2x1^2 + b3x1^3 ....... + bnxn^n
"""

# The new employee says he has 20+yrs experience and previously earned 160k salary and hence asking for more than 160k
#HR calls the new employees previous employer and manages to get some data 
# so the new employee was a regional manager so as per the data set he is at level 6.5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('OldCompanyData.csv')

# X = dataset.iloc[:,1:2].values

X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,2].values

# fitting linear regresssion to the dataset

from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()
lin_reg.fit(X,y)

# fitting the Polynomial Regression to the dataset
# in X_poly 1st column will be the constant for b0 , 2nd column is the value of X, third column is square of X as degree is set to 2

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)   # first start with degree 2
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualization of Linear Regression and check the salary for 6.5 level on graph

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Linear Regression ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

plt.scatter(X, y, color = 'red')
#plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.plot(X, lin_reg_2.predict(X_poly),color = 'blue')
plt.title("Polynomial Regression ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# for higher resolution and continuos curve we increament the levels by 0.1
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.title("Polynomial Regression ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
























