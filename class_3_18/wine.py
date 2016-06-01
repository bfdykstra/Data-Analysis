# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:28:37 2016

@author: benjamindykstra
"""

import pandas as pd
df_wine = pd.read_csv(
'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)

#seperate the data into training and test sets

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
x,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, 
                                                    random_state = 0)
                                                    
#scale data
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

import numpy as np
cov_mat = np.cov(x_train_std.T)

#obtain eigenvalues, eigenvectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n %s' %eigen_vals)

#compute variance explained ratio
total = sum(eigen_vals)
var_exp = eigen_vals/total  #this line is not correct!!!!
cum_var_exp = np.cumsum(var_exp) #calculate the cum sum of explained variance

import matplotlib.pyplot as plt
plt.bar(range(1,14), var_exp, alpha = 0.5, align = 'center', 
        label = 'individual explained variance')
plt.step(range(1,14), cum_var_exp, where = 'mid', 
         label = 'cumulative explained variance')

plt.ylabel('explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc = 'best')
plt.show()


#feature transformation
#sort eigenpairs by decreasing order of the eigenvals
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) 
for i in range(len(eigen_vals))]
    
eigen_pairs.sort(reverse = True)

#collect two eigenvectors that correspond to the two largest values to capture
# about 60% of the variance
w = np.hstack((eigen_pairs[0][1][:,np.newaxis], eigen_pairs[1][1][:,np.newaxis]))
print'Matrix W: \n', w    #a 13x2 projection matrix

#transform the sample x onto the PCA subspace to get 
#a new vector x' (a 1x2 vector)
x_train_std[0].dot(w)

#transform the entire 124x13-dimensional training set
x_train_pca = x_train_std.dot(w)

colors = ['r','b', 'g']
markers = ['s','x','o']
for l,c,m in zip(np.unique(y_train), colors, markers):
    plt.scatter(x_train_pca[y_train == l, 0],
                x_train_pca[y_train == l, 1],
c = c, label = l, marker = m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.show()


#now doing pca with scikit-learn
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
pca = PCA(n_components = 2)
knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
x_train_pca_sklearn = pca.fit_transform(x_train_std)
x_test_pca_sklearn = pca.fit_transform(x_test_std)
knn.fit(x_train_pca_sklearn, y_train)

y_pred_sklearn = knn.predict(x_test_pca_sklearn)
print 'Misclassified samples: %d' %(y_test != y_pred_sklearn).sum()