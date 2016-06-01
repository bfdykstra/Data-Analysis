# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:26:27 2016

@author: benjamindykstra
"""

from Perceptron import Perceptron
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None )
df.tail()


#extract data corresponding to 50 iris setosa and 50 iris veriscolor flowers
# and convert into 2 class labels 1 (versicolor) and -1 (setosa). Create 
#scatterplot

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1,1)
x = df.iloc[0:100, [0,2]].values
plt.scatter(x[:50, 0], x[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal lenght')
plt.legend(loc = 'upper left')
plt.show()

#train perceptron alg on dataset

ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(x, y)
plt.plot(range(1, len(ppn.errors_)+ 1), ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

