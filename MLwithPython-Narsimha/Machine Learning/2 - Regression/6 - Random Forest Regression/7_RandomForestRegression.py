# -*- coding: utf-8 -*-
"""
Random Forest Regression 
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("E:\\MLData\\Position_Salaries.csv")

X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
                                  # start with 10 tree and go upto 300 trees
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)

y_pred = regressor.predict(6.5)

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y, color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Truth or Bluff")
plt.show()
