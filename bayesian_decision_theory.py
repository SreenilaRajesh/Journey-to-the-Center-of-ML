# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:09:16 2019

@author: 17pd32
"""
#Bayesian Decision Theory
#Given a set of documents, each of which is related to either Spam or Ham
#And it is represented as an eight bit binary vector with wi = 1 or 0;1 means the word wi occurs in the document 0 means not.
#The problem is modelled using Bayesian decision theory to obtain the optimal classification rule.
#training set T is given in csv - first five is Spam, remaining Ham.
#The test data is predicted considering the losses

import pandas as pd
import numpy as np
df=pd.read_csv("sh.csv",header=None)

features = df.values[:,:8]
target = df.values[:,8]
domf=[2]*8
target=list(target)
tfeatures=features.T
unique_elements=[]
classes=[1,0]

#a list should be maintained to have different classes
for i in range(len(domf)):#no of feature is 4# or use len(domf)
    unique_elements.insert(i,list(np.unique(tfeatures[i])))

#conditional probability
def condprob(fj,vm,k):
    nk=target.count(k)#no of datapoints in kth class
    nmk=0
    for i in range(len(tfeatures[fj])):
        if(tfeatures[fj][i]==vm and target[i]==k):
            nmk=nmk+1
    return nmk,nk

#computing the conditional probability table
dict1={}
for i in range(len(domf)):#no of features is 4
    dict1[i]={}
    for j in unique_elements[i]:
        dict1[i][j] = {}
        #here another loop k to iterate through different classes
        for k in classes:
            nmk,nk=condprob(i,j,k)
            dict1[i][j][k] =nmk/nk
print(dict1)

#laplacian smoothening 
def lapsmooth(i,k):#i is the features, j is different values that feature take,k is the class
    for j in dict1[i]:
         nmk,nk=condprob(i,j,k)
         dict1[i][j][k]=(nmk+1/domf[i])/nk+1
for i in range(len(domf)):#finding whether the smoothening has to be done
    for j in dict1[i]:
        for k in dict1[i][j]:
            if(dict1[i][j][k]==0.0):
                lapsmooth(i,k)#smoothing is done where there's zero
print(dict1)

#1st row - loss for actual spam predicted as spam and ham
#2nd row - loss for actual ham predicted as spam and ham
lm=[[0,0.5],[0.2,0]]
#pi=[1,0,0,1,1,1,0,1]#instead input the list of test data
pi=[0,1,1,0,1,0,1,0]
#iterate through different classes
prevprob=0
totprob=0
for k in classes:
    prob=target.count(k)/len(target)
    for i in range(len(domf)):    
        prob=prob*dict1[i][pi[i]][k]
    totprob=totprob+prob
probl=[]
for k in classes:
    prob=target.count(k)/len(target)
    for i in range(len(domf)):    
        prob=prob*dict1[i][pi[i]][k]
    prob=prob/totprob
    probl.append(prob)
print(probl)
#risk calculation 
riskl=[]
minimum=1
for k in classes:#k is the prediction
    summ=0
    for j in classes:#j actual
        summ=summ+probl[j]*lm[j][k]
    riskl.append(summ)
    if(summ<minimum):
        minimum=summ
        out=k
#based on the data display the out
print(riskl)
if(out==1):
    print("Spam")
else:
    print("Ham")

    


