# -*- coding: utf-8 -*-
"""

Decision Tree
"""
# Decision Tree Regression Tree

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("OldCompanyData.csv")

X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,2].values

# Fitting the decision regression tree to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict(6.5)

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Truth or Bluff")
plt.show()
