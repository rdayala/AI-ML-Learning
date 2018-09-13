# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("SUV_Ads.csv")

# matrix of independent variables X and vector of dependent variables y
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# split dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# feature scaling of age and salary column
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fit the K-NN classifier into the object
# as Knn works on Euclidean distance hence we choose minkowski and with power parameter 2, as Euclidien power parameter is 2
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

# predictions based on X_test the test set
y_pred = classifier.predict(X_test)

# confusion matix to findout the number of correct predictions and incorrect predictions

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# test set visual

from matplotlib.colors import ListedColormap

X_set, y_set = X_test,y_test

X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,1].max()+1,step=0.01),
                     np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j , 0], X_set[y_set == j, 1],
                c=ListedColormap(('red','green'))(i), label = j)
plt.title('KNN (SUV Purchase)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
















