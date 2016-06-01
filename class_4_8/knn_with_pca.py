# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:26:15 2016

@author: benjamindykstra
"""

import pandas as pd
train = pd.read_csv("~/Documents/Math352/hw4/train.csv")

#seperate the data into training and test sets

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
x_train = train.iloc[:,1:].values
                                                    


#determine number of principal components
def principal_comps(x, variance_retained):
    #scale data
    sc = StandardScaler()
    #x_train_std = sc.fit_transform(x)
    
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components = len(x.T))
    x_train_pca = pca.fit_transform(x_train_std)
    cum_VE = 0
    i = 1
    while cum_VE < variance_retained:
        i= i + 1
        cum_VE = sum(pca.explained_variance_ratio_[0:i])
        npcs = i
    #apply principal components
    print()
    print "Use " + str(npcs) + " principal components to retain " + str(variance_retained*100) + "% of the variance"
    pca = PCA(n_components = npcs)

principal_comps(x_train, .90)
principal_comps(x_train, .95)
principal_comps(x_train, .99)
