# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#dimensionality reduction using Principal Component Analysis in IRIS Dataset
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df=pd.read_csv("IRIS.csv")
features = df.values[:,:4]
target = df.values[:,4]
features= features.astype('float64')
Tf=features.T

#reducing the dimensions using PCA
CovMat=np.cov(Tf)
w, v = LA.eig(CovMat)
zipped = zip(w, v)
list1, list2 = zip(*sorted(zip(w, v),reverse=True))
eigenvec=np.asarray(list2[:2])
z= np.dot(eigenvec,Tf)

#Checking the accuracy scores for NaiveBayes Classifier 
#when trained and tested using original and reduced data
X_train, X_test, y_train, y_test = train_test_split(z.T,target, test_size=0.33, random_state=42)
nb=GaussianNB()
nb.fit(X_train,y_train)
ypred=nb.predict(X_test)
print("2features")
print(accuracy_score(y_test,ypred))


X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.33, random_state=42)
nb.fit(X_train,y_train)
ypred=nb.predict(X_test)
print("4features")
print(accuracy_score(y_test,ypred))


