# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:10:20 2016

@author: benjamindykstra
"""

import pandas as pd
import matplotlib as plt
import statsmodels.api as sm
import numpy as np


train = pd.read_csv("Problem8_train.csv")
test = pd.read_csv("Problem8_test.csv")

#want to predict revenue
x_train = train.values[:,0:37]#all values except for revenue
y_train = train[['revenue']].values #revenue
x_test = test.values[:,1:38]

#add some constants so that we have a theta0
x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

#fit the model to our data
result = sm.OLS(y_train, x_train).fit()

print(result.summary())

y_pred = result.predict(x_test)
#np.savetxt("Problem8_submission.csv", y_pred, delimiter = ",")
