# -*- coding: utf-8 -*-
"""
Created on Sun May 13 00:36:47 2018

@author: NILESH
"""
import csv
import os
import random
import numpy as np
from sklearn import svm
import copy
#Divided the data set as 65%-20%-15%

def accuracy(predected,actual):
    length = len(predected)
    count = 0
    for i in range(0,length):
        if(predected[i]==actual[i]):
            count = count + 1
    return ((count*100)/length)    

def feature_scaling(train_X):
    train_X_scaled = copy.deepcopy(train_X)
    number_of_instances, number_of_features = train_X_scaled.shape
    mean_for_instances = []
    std_for_instances = []
    for i in range(0,number_of_features):
        mean_for_instances.append(np.mean(train_X_scaled.T[i]))
        std_for_instances.append(np.std(train_X_scaled.T[i]))
        train_X_scaled.T[i] -= mean_for_instances[i]
        train_X_scaled.T[i] /= std_for_instances[i]
    return mean_for_instances, std_for_instances, train_X_scaled

def scale_feature_using_std_mean(instances,std_array,mean_array):
    length = len(instances[0])
    for i in range(0,length):
        instances.T[i] -= mean_array[i]
        instances.T[i] /= std_array[i]
    return instances    
def create_X_Y(feature_array_with_label_random):
    train_X =[]
    train_Y =[]
    corv_X = []
    corv_Y = []
    test_X = []
    test_Y = []
    length = len(feature_array_with_label_random)
    feature_len = len(feature_array_with_label_random[0])
    t_len = int(np.floor(length*0.70))
    c_len = int(np.floor(length*0.20))
    for i in range(0,t_len):
        train_X.append(feature_array_with_label_random[i][0:feature_len-1])
        train_Y.append(feature_array_with_label_random[i][feature_len-1])
    for i in range(t_len,t_len + c_len):
        corv_X.append(feature_array_with_label_random[i][0:feature_len-1])
        corv_Y.append(feature_array_with_label_random[i][feature_len-1])
    for i in range(t_len + c_len,length):
        test_X.append(feature_array_with_label_random[i][0:feature_len-1])
        test_Y.append(feature_array_with_label_random[i][feature_len-1])
    return np.asarray(train_X,dtype=np.float64),np.asarray(train_Y,dtype=np.float64),np.asarray(corv_X,dtype=np.float64),np.asarray(corv_Y,dtype=np.float64),np.asarray(test_X,dtype=np.float64),np.asarray(test_Y,dtype=np.float64)    

file_path = os.getcwd() + '\dataset_800.csv'
feature_array_with_label = []
with open(file_path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        size = len(row)
        if(size>0):
            feature_array_with_label.append(row)
length = len(feature_array_with_label)
feature_array_with_label1 = feature_array_with_label[1:length]
random.shuffle(feature_array_with_label1)
train_X,train_Y,corv_X,corv_Y,test_X,test_Y = create_X_Y(feature_array_with_label1)            
mean_for_trainX,std_for_train_X,scaled_train_X = feature_scaling(train_X)
#classifier = svm.SVC(kernel='rbf',C=1,gamma=1)
#classifier.fit(scaled_train_X,train_Y)
#predictions = classifier.predict(scaled_train_X) 
#print(accuracy(predictions,train_Y))
scaled_cross_validation = scale_feature_using_std_mean(corv_X,std_for_train_X,mean_for_trainX)
#predictions_cor = classifier.predict(scaled_cross_validation)
#print(accuracy(predictions_cor,corv_Y))
#********************Hyperparameter_selection********************
#scaled_train_X = (train_X > 0) * 1
#scaled_cross_validation = (corv_X >0) * 1
#scaled_test_X = (test_X > 0) * 1
hyper_parameters_gamma = [1000,500,250,125,75,40,25,10,1]
hyper_parameters_C = [0.001,0.03,0.1,0.3,1,3,10,30]
len_gamma =len(hyper_parameters_gamma)
len_c = len(hyper_parameters_C)
max_val = 0
C_1 = 0
Gamma_1 = 0
for i in range(0,len_c):
    for j in range(0,len_gamma):
        svm_classifier =  svm.SVC(kernel='rbf',C=hyper_parameters_C[i],gamma=hyper_parameters_gamma[j])
        svm_classifier.fit(scaled_train_X,train_Y)
        predected_class = svm_classifier.predict(scaled_cross_validation)
        val = accuracy(predected_class,corv_Y)
        print(val)
        if(max_val<val):
            max_val = val
            C_1 = hyper_parameters_C[i]
            Gamma_1 = hyper_parameters_gamma[j]
            
scaled_test_X = scale_feature_using_std_mean(test_X,std_for_train_X,mean_for_trainX)
svm_classifier =  svm.SVC(kernel='rbf',C=C_1,gamma=Gamma_1)
svm_classifier.fit(scaled_train_X,train_Y)
predected_class = svm_classifier.predict(scaled_test_X)          
print(accuracy(predected_class,test_Y))
  