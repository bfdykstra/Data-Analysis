# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:22:23 2016

@author: benjamindykstra
"""

#problem 4 

#use all features to make a prediction using ridge regression

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("Problem8_train.csv")
test = pd.read_csv("Problem8_test.csv")

#specify features 
x_train = train.values[:,0:37]#all values except for revenue
y_train = train[['revenue']].values #revenue
x_test = test.values[:,1:38]

#add some constants so that we have a theta0
x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

#scale paramters
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

#ridge regression using lamdas .01, .05, .1, 1, 5, 10, 25, 50
RR = linear_model.RidgeCV(alphas = [0.01, 0.05, 0.1, 1, 5, 10, 25, 50])
RR.fit(x_train_std, y_train)
best_lambda1 = RR.alpha_
print
print "The best lambda for Ridge Regression: ", best_lambda1
print
#run model on the test set
y_test_pred_ridge = RR.predict(x_test_std)


#export_csv = {"ID": pd.Series(range(0,38)) , "revenue": y_test_pred_ridge}

np.savetxt("Problem4_submission.csv", y_test_pred_ridge, delimiter = ",")

#Lasso regression on scaled data
lasso = linear_model.LassoCV(alphas = [.01, .05, .1, 1, 5, 10, 25, 50])
lasso.fit(x_train_std, y_train)
best_lambda2 = lasso.alpha_
print
print "The best lasso lambda ", best_lambda2
print
#run it again on test set
y_test_pred_lasso = lasso.predict(x_test_std)

np.savetxt("Problem4_lasso_submission.csv", y_test_pred_lasso, 
           delimiter = ",")


#ElasticNet regression on scaled data
elastic = linear_model.ElasticNetCV(alphas = [.01, .05, .1, 1, 
                                              5, 10, 25, 50])
elastic.fit(x_train_std, y_train)
best_lambda2 = elastic.alpha_
print
print"The best lambda ", best_lambda2
print
#run it again on test set
y_test_pred_elastic = elastic.predict(x_test_std)

np.savetxt("Problem4_elastic_submission.csv", y_test_pred_elastic, 
           delimiter = ",")






