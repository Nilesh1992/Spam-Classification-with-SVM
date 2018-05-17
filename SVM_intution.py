# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:56:34 2018

@author: NILESH
"""
from sklearn import svm
import scipy.io as spio
import os
import numpy as nu
import matplotlib.pyplot as plt

#***************************************************************

def get_coordinate(intercept,value):
    return (intercept*(-1))/value   

file_path = os.getcwd() + "\ex6data1.mat"
mat = spio.loadmat(file_path, squeeze_me=True) #This is to load the matrix
X = mat['X']
Y = mat['y']
length = len(X)
#plt.plot(X.T[0],X.T[1],'ro')
#************************** C = 1 ******************************
svm_classifier  = svm.SVC(kernel='linear',C=1)
svm_classifier.fit(X,Y)
coeff = svm_classifier.coef_
intercept = svm_classifier.intercept_
x1 = get_coordinate(intercept,coeff[0][0])
y2 = get_coordinate(intercept,coeff[0][1])
#plt.plot([x1,0],[0,y2],'k-')
#************************** C = 50 ******************************
svm_classifier  = svm.SVC(kernel='linear',C=10)
svm_classifier.fit(X,Y)
coeff = svm_classifier.coef_
intercept = svm_classifier.intercept_
x1 = get_coordinate(intercept,coeff[0][0])
y2 = get_coordinate(intercept,coeff[0][1])
#plt.plot([x1,0],[0,y2],'k-')
#************************** C = 100******************************
svm_classifier  = svm.SVC(kernel='linear',C=100)
svm_classifier.fit(X,Y)
coeff = svm_classifier.coef_
intercept = svm_classifier.intercept_
x1 = get_coordinate(intercept,coeff[0][0])
y2 = get_coordinate(intercept,coeff[0][1])
#plt.plot([x1,0],[0,y2],'k-')
#***************************************************************************
#Dataset-2
#***************************************************************************
def get_gamma(sigma):
    return ((1/(2*nu.square(sigma))))

file_path = os.getcwd() + "\ex6data2.mat"
mat = spio.loadmat(file_path, squeeze_me=True) #This is to load the matrix
X_1 = mat['X']
Y_1 = mat['y']
length = len(X_1)
positive_class = nu.array([[0,0]],dtype=nu.float64)
negative_class = nu.array([[0,0]],dtype=nu.float64)
number_of_traning_data = len(Y_1)
print(number_of_traning_data)
for i in range(0,number_of_traning_data-1):
    if(Y_1[i]==1.0):
        positive_class = nu.append(positive_class,[X_1[i]],axis=0)
    else:
        negative_class = nu.append(negative_class,[X_1[i]],axis=0)
data_x_axis = negative_class.T[0][1::] #negative_class.T[0][1::] this give use the data in range 1 to m
data_y_axis = negative_class.T[1][1::]
plt.plot(data_x_axis,data_y_axis,'yo')
data_x_axis = positive_class.T[0][1::] #positive_class.T[0][1::] this give use the data in range 1 to m
data_y_axis = positive_class.T[1][1::]
plt.plot(data_x_axis,data_y_axis,'r+')

#plt.plot(X_1.T[0],X_1.T[1],'ro')
# C=1 and sigma = 0.1 
#******************************************************
gam= get_gamma(0.1)
svm_classifier  = svm.SVC(kernel='rbf', C=1, gamma=gam)
svm_classifier.fit(X_1,Y_1)
minimum_1 = nu.min(X_1.T[0]),nu.min(X_1.T[1])
maximum_1 = nu.max(X_1.T[0]),nu.max(X_1.T[1])
minimum = nu.min(minimum_1)
maximum = nu.max(maximum_1)
XX, YY = nu.meshgrid(nu.arange(minimum,maximum,0.01), nu.arange(minimum,maximum,0.001))
Z = svm_classifier.predict(nu.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
plt.contourf(XX, YY, Z, cmap=plt.cm.Paired, alpha=0.5)

# C=10 and sigma = 0.1 
#******************************************************
gam= get_gamma(0.1)
svm_classifier  = svm.SVC(kernel='rbf', C=10, gamma=gam)
svm_classifier.fit(X_1,Y_1)
minimum_1 = nu.min(X_1.T[0]),nu.min(X_1.T[1])
maximum_1 = nu.max(X_1.T[0]),nu.max(X_1.T[1])
minimum = nu.min(minimum_1)
maximum = nu.max(maximum_1)
XX, YY = nu.meshgrid(nu.arange(minimum,maximum,0.01), nu.arange(minimum,maximum,0.001))
Z = svm_classifier.predict(nu.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
plt.contourf(XX, YY, Z, cmap=plt.cm.Paired,alpha=0.5)

# C=100 and sigma = 0.1 
#******************************************************
gam= get_gamma(0.1)
svm_classifier  = svm.SVC(kernel='rbf', C=100, gamma=gam)
svm_classifier.fit(X_1,Y_1)
minimum_1 = nu.min(X_1.T[0]),nu.min(X_1.T[1])
maximum_1 = nu.max(X_1.T[0]),nu.max(X_1.T[1])
minimum = nu.min(minimum_1)
maximum = nu.max(maximum_1)
XX, YY = nu.meshgrid(nu.arange(minimum,maximum,0.01), nu.arange(minimum,maximum,0.001))
Z = svm_classifier.predict(nu.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
plt.contourf(XX, YY, Z, cmap=plt.cm.Paired,alpha=0.5)

