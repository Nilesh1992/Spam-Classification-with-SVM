# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:38:54 2018

@author: NILESH
"""

from sklearn import svm
import scipy.io as spio
import os
import numpy as nu
import matplotlib.pyplot as plt
def get_cross_validation_predection_accurcy(predected_class, actual_class):
    accuracy = 0
    len_of_set = len(predected_class)
    for i in range(0,len_of_set-1):
        if(predected_class[i]==actual_class[i]):
            accuracy = accuracy + 1
    return (accuracy/len_of_set)        

file_path = os.getcwd() + "\ex6data3.mat"
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
data_x_axis_1 = positive_class.T[0][1::] #positive_class.T[0][1::] this give use the data in range 1 to m
data_y_axis_1 = positive_class.T[1][1::]
plt.plot(data_x_axis_1,data_y_axis_1,'r+')
hyper_parameters_gamma = [1000,500,250,125,75,40,25,10,1]
hyper_parameters_C = [0.001,0.03,0.1,0.3,1,3,10,30]
len_gamma =len(hyper_parameters_gamma)
len_c = len(hyper_parameters_C)
cos_X = mat['Xval']
len_of_cross = len(cos_X)
actual_class = mat['yval']
predected_accuracy = []
max_val = 0
C_1 = 0
Gamma_1 = 0
minimum_1 = nu.min(X_1.T[0]),nu.min(X_1.T[1])
maximum_1 = nu.max(X_1.T[0]),nu.max(X_1.T[1])
minimum = nu.min(minimum_1)
maximum = nu.max(maximum_1)
XX, YY = nu.meshgrid(nu.arange(minimum,maximum,0.01), nu.arange(minimum,maximum,0.001))        
for i in range(0,len_c):
    for j in range(0,len_gamma):
        #print("C = %f and gamma = %f"%(hyper_parameters_C[i],hyper_parameters_gamma[j]))
        plt.figure()
        plt.title('C='+str(hyper_parameters_C[i])+'Gamma='+str(hyper_parameters_gamma[j]))
        plt.plot(data_x_axis,data_y_axis,'yo')
        plt.plot(data_x_axis_1,data_y_axis_1,'r+')
        svm_classifier =  svm.SVC(kernel='rbf',C=hyper_parameters_C[i],gamma=hyper_parameters_gamma[j])
        svm_classifier.fit(X_1,Y_1)
        predected_class = svm_classifier.predict(cos_X)
        val = get_cross_validation_predection_accurcy(predected_class,actual_class)
        if(max_val<val):
            max_val = val
            C_1 = hyper_parameters_C[i]
            Gamma_1 = hyper_parameters_gamma[j]
        Z = svm_classifier.predict(nu.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.contourf(XX, YY, Z, cmap=plt.cm.Paired,alpha=0.5)
        predected_accuracy.append(val)

X_axis = nu.arange(0,56,1)
svm_classifier =  svm.SVC(kernel='rbf',C=C_1,gamma=Gamma_1)
svm_classifier.fit(X_1,Y_1)
plt.figure()
minimum_1 = nu.min(X_1.T[0]),nu.min(X_1.T[1])
maximum_1 = nu.max(X_1.T[0]),nu.max(X_1.T[1])
minimum = nu.min(minimum_1)
maximum = nu.max(maximum_1)
XX, YY = nu.meshgrid(nu.arange(minimum,maximum,0.01), nu.arange(minimum,maximum,0.001))
Z = svm_classifier.predict(nu.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
plt.contourf(XX, YY, Z, cmap=plt.cm.Paired,alpha=0.5)
plt.figure()
plt.plot(X_axis,predected_accuracy,'ro')
