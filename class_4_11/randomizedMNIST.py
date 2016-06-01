# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:07:18 2016

@author: benjamindykstra
"""


import numpy as np
from scipy.special import expit
import pandas as pd

x = pd.read_csv("~/Documents/Math352/hw4/test.csv")
#train = pd.read_csv("~/Documents/Math352/hw4/train.csv")

n_hidden = 30
n_features = x.shape[1]
n_output = 10

#randomly generate the weight matrix
# - w1: input to hedden weight matrix
# - w2: is the hidden to output weight matrix

w1 = np.random.uniform(-1.0, 1.0, size = n_hidden*(n_features + 1))
w1 = w1.reshape(n_hidden, n_features + 1)
w2 = np.random.uniform(-1.0, 1.0, size = n_output*(n_hidden + 1))
w2 = w2.reshape(n_output, n_hidden + 1)

#add bias unit: input to hidden
x_new = np.ones((x.shape[0], x.shape[1] + 1))
x_new[:, 1:] = x
a1 = x_new
z2 = w1.dot(a1.T)
a2 = expit(z2)

#add bias unit: hidden to output
a2_new = np.ones((a2.shape[0] + 1, a2.shape[1]))
a2_new[1:, :] = a2
z3 = w2.dot(a2_new)
#every column of a3 is the probability vector
a3 = expit(z3)

y_pred = np.argmax(a3, axis = 0)
np.savetxt("MNISTrandomized.csv", y_pred, delimiter = ",")