# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 22:21:35 2016

@author: benjamindykstra
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

#5a
test = pd.read_csv("~/Documents/Math352/hw4/test.csv")
train = pd.read_csv("~/Documents/Math352/hw4/train.csv")

x_train, y_train = train.iloc[:,1:].values, train.iloc[:,0].values
x_test = test.iloc[:,0:].values


#scale data
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

import numpy as np
cov_mat = np.cov(x_train_std.T)

#obtain eigenvalues, eigenvectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)


#compute variance explained ratio
total = sum(eigen_vals)
var_exp = eigen_vals/total  
eigen_vals.sort()
cum_var_exp = np.cumsum(var_exp) #calculate the cum sum of explained variance

"""
#now doing pca with scikit-learn
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

#90% variance explained
pca = PCA(n_components = 229)
knn = KNeighborsClassifier(n_neighbors = 3, p = 2, metric = 'minkowski')
x_train_pca_sklearn = pca.fit_transform(x_train_std)
x_test_pca_sklearn = pca.fit_transform(x_test_std)
knn.fit(x_train_pca_sklearn, y_train)

y_pred_sklearn = knn.predict(x_test_pca_sklearn)

knn_pred = pd.DataFrame({'id' : range(1,28001), 'label' : y_pred_sklearn}) #puts data into dataframe
np.savetxt("KNN_pred_problem5_90.csv",knn_pred, delimiter = ",")


#95% variance explained
pca = PCA(n_components = 320)
knn = KNeighborsClassifier(n_neighbors = 3, p = 2, metric = 'minkowski')
x_train_pca_sklearn = pca.fit_transform(x_train_std)
x_test_pca_sklearn = pca.fit_transform(x_test_std)
knn.fit(x_train_pca_sklearn, y_train)

y_pred_sklearn = knn.predict(x_test_pca_sklearn)

knn_pred = pd.DataFrame({'id' : range(1,28001), 'label' : y_pred_sklearn}) #puts data into dataframe
np.savetxt("KNN_pred_problem5_95.csv",knn_pred, delimiter = ",")

#99% variance explained
pca = PCA(n_components = 558)
knn = KNeighborsClassifier(n_neighbors = 3, p = 2, metric = 'minkowski')
x_train_pca_sklearn = pca.fit_transform(x_train_std)
x_test_pca_sklearn = pca.fit_transform(x_test_std)
knn.fit(x_train_pca_sklearn, y_train)

y_pred_sklearn = knn.predict(x_test_pca_sklearn)

knn_pred = pd.DataFrame({'id' : range(1,28001), 'label' : y_pred_sklearn}) #puts data into dataframe
np.savetxt("KNN_pred_problem5_99.csv",knn_pred, delimiter = ",")
"""


"""Function that does knn and pca on data
    num_components = how many components to use
    num_neighbors = # of neighbors in knn
    per_explained = % of variation explained with the data
"""
def pca_knn(num_components, num_neighbors, x_train, x_test, y_train, per_explained):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    
    pca = PCA(n_components = num_components)
    knn = KNeighborsClassifier(n_neighbors = num_neighbors, p = 2, metric = 'minkowski')
    x_train_pca_sklearn = pca.fit_transform(x_train)
    x_test_pca_sklearn = pca.transform(x_test_std)
    knn.fit(x_train_pca_sklearn, y_train)

    y_pred_sklearn = knn.predict(x_test_pca_sklearn)
    
    print "about to write to file for %s " + str(per_explained) + "%"
    #knn_pred = pd.DataFrame({'id' : range(1,28001), 'label' : y_pred_sklearn}) #puts data into dataframe
    file_label = "KNN_pred_problem5_" + str(per_explained) + ".csv"
    np.savetxt(file_label, y_pred_sklearn, delimiter = ",")


    
pca_knn(num_components = 230, num_neighbors = 3, x_train = x_train_std, 
        x_test = x_test_std, y_train = y_train, per_explained = 90)
pca_knn(num_components = 321, num_neighbors = 3, x_train = x_train_std, 
        x_test = x_test_std, y_train = y_train, per_explained = 95)
pca_knn(num_components = 559, num_neighbors = 3, x_train = x_train_std, 
        x_test = x_test_std, y_train = y_train, per_explained = 99)
    

