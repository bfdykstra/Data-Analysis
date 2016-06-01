# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:45:30 2016

@author: benjamindykstra
"""

#get the data
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'

df = pd.read_csv(url, header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
              
#specify features and the response
x = df[['CRIM', 'ZN', 'LSTAT', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B',]].values
y = df['MEDV'].values

#split data into training and testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, 
                                                    random_state = 1)

#ridge regression on unscaled data
RR = linear_model.RidgeCV(alphas = [0.01, 0.05, 0.1])
RR.fit(x_train, y_train)
best_lambda1 = RR.alpha_
print
print(best_lambda1)
print
#run model on the test set
y_test_pred = RR.predict(x_test)

#score the result
mse1 = mean_squared_error(y_test, y_test_pred)
print(mse1)

#scale parameters
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
#y_train_std = sc.transform(y_train)
x_test_std = sc.transform(x_test)


#ridge regression on scaled data
RR2 = linear_model.RidgeCV(alphas = [.01, .05, .1])
RR2.fit(x_train_std, y_train)
best_lambda2 = RR2.alpha_
print
print("The best lambda ", best_lambda2)
print
#run it again on test set
y_test_pred = RR2.predict(x_test_std)
mse2 = mean_squared_error(y_test, y_test_pred)
print "mean squared error for ridge: ", mse2



#Lasso regression on scaled data
lasso = linear_model.LassoCV(alphas = [.01, .05, .1])
lasso.fit(x_train_std, y_train)
best_lambda2 = lasso.alpha_
print
print "The best lambda ", best_lambda2
print
#run it again on test set
y_test_pred = lasso.predict(x_test_std)
mse2 = mean_squared_error(y_test, y_test_pred)
print "mean squared error for lasso: ", mse2 


#ElasticNet regression on scaled data
elastic = linear_model.ElasticNetCV(alphas = [.01, .05, .1])
elastic.fit(x_train_std, y_train)
best_lambda2 = elastic.alpha_
print
print("The best lambda ", best_lambda2)
print
#run it again on test set
y_test_pred = elastic.predict(x_test_std)
mse2 = mean_squared_error(y_test, y_test_pred)
print "mean squared error for elastic: ", mse2
