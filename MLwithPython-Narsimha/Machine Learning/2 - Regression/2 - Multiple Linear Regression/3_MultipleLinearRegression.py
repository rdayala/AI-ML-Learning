# -*- coding: utf-8 -*-
"""

Multiple linear regression

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Companies.csv')


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding Categorical variable

labelEncoderObj_X = LabelEncoder()
X[:,3] = labelEncoderObj_X.fit_transform(X[:,3])    # converts the value into number and then OneHotEncoder will be able to encode it

# one Hot encoder will convert the country column into 3 dummy columns 
onehotencoder = OneHotEncoder(categorical_features = [3] )
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap, dummy variable trap occurs when we have multiple dummy columns
# for a single column which the regression model gets confused with as the values encoded for the categories is 0 or 1
# hence to avoid issues in the end result we need to remove one of the dummy e.g. 3 dummy columns then use 2, 2 dummy columns then use 1 

# though the library takes care of this but just an explicit step as precation that we dont get trapped
# this step can be skipped for this example 
X = X[:,1:]

# Training and testing split

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


# Building the optimal model by Backward elimination
# As per the linear regression formula  y = b0 + b1*x1 + b2*x2 ..... + bn*xn
# b0 (intercept is not included by default) the constant cannot be used individually in backward elimination process hence we need to introduce x0
# hence using np.append we will add a new column at the start of the X matrix containg all 1's


import statsmodels.formula.api as sm

# arr is the 1st column containing 1 will be created and it that we will fit the X matrix using values parameter axis 0 row 1 column 
X = np.append(arr = np.ones((50,1)).astype(int) ,
                      values = X, axis = 1)

# X_opt is a new variable which will contain the optimal list of independent variables which influence the profit value (dependent variables)
X_opt = X[:,[0,1,2,3,4,5]]

# now to start the elimination we will have to calculate the p value of each dependent variable
# we set the significance level to 0.05 (5%) any independent variable whose p value is less than 5% will stay in the model
# and for those whose p value is more that 5% will get eliminated from the X matrix

# OLS ordinary least square model

            #const, D1,D2,RD,Admin,Mkt
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# at this step we removed index 2 from X matrix as its p values was 0.990 which is more than SL 
# repeat this step until you get the independent variable set whose p value is less tha SL
# endog (dependent variable)
#exog is the matrix of features
		#const, D1,RD,Admin,Mkt
X_opt = X[:,[0,1,3,4,5]]    # just a copy of X matrix of features 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()
        # const,RD,Admin,Mkt
X_opt = X[:,[0,3,4,5]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

          # const,RD,Mkt
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

















