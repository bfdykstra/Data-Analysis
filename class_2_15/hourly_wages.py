# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:27:30 2016

@author: benjamindykstra
"""

import pandas as pd
import matplotlib as plt
import statsmodels.api as sm
import numpy as np

##predict hourly wages from age and sex
train = pd.read_csv("Income_training.csv")
test = pd.read_csv("Income_testing.csv")

#create training and test data sets from variables
x_train = train[['age', 'yearsEducation', 'sex1M0F']].values
y_train = train[['compositeHourlyWages']].values
x_test = test[['age', 'yearsEducation', 'sex1M0F']].values



#tells python to use the model
# y = theta0 + theta1*x1 + theta2*x2 + theta3*x2
x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

#fit the model to our data 
result = sm.OLS(y_train, x_train).fit()

#summarize the model in a table 
print(result.summary())

y_pred = result.predict(x_test)

np.savetxt("OLS_pred.csv",y_pred, delimiter = ",")