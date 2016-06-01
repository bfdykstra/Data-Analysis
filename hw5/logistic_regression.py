# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:09:02 2016

@author: benjamindykstra
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#5a
test = pd.read_csv("~/Documents/Math352/hw5/test.csv")
train = pd.read_csv("~/Documents/Math352/hw5/train.csv")

x_train, y_train = train.iloc[:,1:].values, train.iloc[:,0].values
x_test = test.iloc[:,0:].values


#scale data
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

#run logistic regressions
#C = 1
lr = LogisticRegression(C = 1, random_state = 17)
lr.fit(x_train_std, y_train)
y_pred_lr = lr.predict(x_test_std)
file_label = "log_reg_pred_problem7_" + "C1" + ".csv"
np.savetxt(file_label, y_pred_lr, delimiter = ",")

#C = 100
lr = LogisticRegression(C = 100, random_state = 17)
lr.fit(x_train_std, y_train)
y_pred_lr = lr.predict(x_test_std)
file_label = "log_reg_pred_problem7_" + "C100" + ".csv"
np.savetxt(file_label, y_pred_lr, delimiter = ",")

#C = 10000
lr = LogisticRegression(C = 10000, random_state = 17)
lr.fit(x_train_std, y_train)
y_pred_lr = lr.predict(x_test_std)
file_label = "log_reg_pred_problem7_" + "C10000" + ".csv"
np.savetxt(file_label, y_pred_lr, delimiter = ",")
