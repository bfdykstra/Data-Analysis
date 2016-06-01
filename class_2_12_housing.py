# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:33:17 2016

@author: benjamindykstra
"""

import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df = pd.read_csv(url, header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN','INDUS', 'CHAS','NOX','RM','AGE','DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
              
import statsmodels.api as sm
x_init = df[['RM','CRIM']].values
y = df[['MEDV']].values
x = sm.add_constant(x_init) #Tells python to use theta_0
result = sm.OLS(y, x).fit()
print(result.summary())

##### not adding constant on to x vector

x_init = df[['RM','CRIM']].values
y = df[['MEDV']].values
result = sm.OLS(y, x_init).fit()
print(result.summary())

x = [10, 12, 2, 0, 8, 5]
y = [70, 65, 96, 94, 75, 82]
result = sm.OLS(y, x).fit()
print(result.summary())