# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:54:30 2016

@author: benjamindykstra
"""

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

test = pd.read_csv("~/Documents/Math352/hw2/Problem7_test.csv")
train = pd.read_csv("~/Documents/Math352/hw2/Problem7_train.csv")

x_train = train.values[:,1:785] #leave out first column bc that is the answer
y_train = train.values[:, 0] #gets just the label which is the answer
x_test = test.values[:,1:785]

#show image of the 3rd observation
imgplot = plt.imshow(x_train[3,:].reshape(28,28)) #reshape it into 28X28 picture

knn = KNeighborsClassifier(n_neighbors = 3, p = 2, metric = 'minkowski')

knn.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
#y_pred = knn.predict(x_train) #predict the numbers from x_train
y_pred = knn.predict(x_test) #should be predicting with test values

import numpy as np
knn_pred = pd.DataFrame({'id' : range(1,2001), 'label' : y_pred}) #puts data into dataframe
np.savetxt("KNN_pred.csv",knn_pred, delimiter = ",")

#accuracy_score(y_train, y_pred) #gives percentage of correctly classified samples









