# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:29:55 2016

@author: benjamindykstra
"""

from sklearn import datasets
import sklearn
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()

#sepal length = iris[:, 0]
#sepal width = iris[:,1]
#petal length = iris[:, 2]
#petal width = iris[:, 3]

#9 a) 
x = iris.data[:,1]
y = iris.data[:, 3]
plt.scatter(x,y)

#b)
#x_hat = np.arange(1.5,6,.03) #150 values between 1.5 and 6
y_hat = -.75*x + 3.5
plt.plot(x, y_hat)
plt.show()

#c) calculate mse
mse_x = 0
mse_y = 0
for i in range(len(x)):
    #mse_x += (y_hat[i] - x[i])**2
    mse_y += (y_hat[i] - y[i])**2

mse = (mse_y)/150
print(mse)



#10 a)
#target 2 = virginica 
#target 0 = setosa
petal_length = iris.data[:,2]
petal_width = iris.data[:,3]
#plt.scatter(petal_width, petal_length)
plt.scatter(petal_width[iris.target == 2],petal_length[iris.target == 2], color = 'red')
plt.scatter(petal_width[iris.target == 0],petal_length[iris.target == 0], color = 'blue')

#b)
plt.plot(petal_width, -petal_width + 4, color = 'black')
plt.show()
#CE = 0

#c)
plt.scatter(petal_width[iris.target == 2],petal_length[iris.target == 2], color = 'red')
plt.scatter(petal_width[iris.target == 0],petal_length[iris.target == 0], color = 'blue')
plt.plot(petal_width, -petal_width + 2, color = 'black')
plt.show()

#calculate classification error
y_hat = -petal_width + 2
y = petal_length[iris.target == 0]
count = 0
for i in range(len(petal_length[iris.target == 0])):
    if y_hat[i] < y[i]:
        count += 1
print(count)
ce = count / len(y)
#CE = 1/10













