# -*- coding: utf-8 -*-
"""
Created on Mon May  7 23:23:34 2018

@author: NILESH
"""
import numpy as nu
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize,word_tokenize
import glob
import operator
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import csv
import os
import gc
all_keywords = {} #This holds all the keywords
def clean_email(raw_html,filename,out):
    cleantext = BeautifulSoup(raw_html, "html.parser").text.lower()
    #file_path  = "\\".join(filename.split("\\")[0:-1])
    #file_name = filename.split("\\")[-1]
    #full_path = file_path + "\\"+ out + "\\" +file_name   
    #file_handle = open(full_path +".txt","w")
    #print(cleantext,file=file_handle)
    #file_handle.write(cleantext)
    #file_handle.close
    #gc.collect()
    cleantext = re.sub(r"[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+","emailaddr",cleantext)
    cleantext = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","httpaddr",cleantext)
    cleantext = re.sub(r"[+-]?\$([0-9]*[.])?[0-9]+","dollermoney",cleantext)
    cleantext = re.sub(r"\d{5}(?:[-\s]\d{4})?$","zipcode",cleantext)
    cleantext = re.sub(r"[,|-]","",cleantext)
    cleantext = re.sub(r"\d+[\/\d. ]*|\d","number", cleantext)
    cleantext = re.sub(r"\%","percent", cleantext)
    cleantext = remove_stop_words(cleantext)
    cleantext = re.sub(r"[.|?|!|'|\"|,|:|;|...|_|/|=|\)|\()|*|^|&|\[|\]]+"," ", cleantext)
    cleantext = re.sub(r"[\n]+"," ", cleantext)
    cleantext = re.sub(r"[-]+"," ", cleantext)
    cleantext = re.sub(r"<.*>","",cleantext)
    cleantext = re.sub(r"[ ]+"," ", cleantext)
    #file_handle = open(full_path +"_remove_clean.txt","w")
    #print(cleantext,file=file_handle)
    #file_handle.write(cleantext)
    #file_handle.close
    #gc.collect() 
    return cleantext

def remove_stop_words(clean_text):
    keywords = word_tokenize(str(clean_text))     
    output = ""
    for w in keywords:
        if (w not in stop_words):
            output = output + " " + w
    return output
def stemming(clean_text,filename,out):
    ps = PorterStemmer()
    words = word_tokenize(clean_text)
    output = ""
    for w in words:
        output = output + ' ' +ps.stem(w)
    #file_path  = "\\".join(filename.split("\\")[0:-1])
    #file_name = filename.split("\\")[-1]
    #full_path = file_path + "\\"+ out + "\\"+ file_name   
    #file_handle = open(full_path +"_potter.txt","w")
    #file_handle.write(output)
    #print(output,file=file_handle)
    #file_handle.close
    #gc.collect()
    return output

def read_email_and_get_features(filename,out):
    try:
        file_handler_1 = open(filename,'r')
        all_lines = file_handler_1.readlines()
        file_handler_1.close
        gc.collect()
        index = all_lines.index("\n")
        raw_html_array =  all_lines[index+1:-1] + [all_lines[-1]]
        raw_html = ' '.join(raw_html_array)
        clean_text = clean_email(raw_html,filename,out).lower()
        final_email_string = stemming(clean_text,filename,out).strip()
        return final_email_string
    except Exception as e:
        print("Could not process the file ")      

def get_keywords(file_list,out):
    keyword_dictonary = {}
    for file in file_list:
        clean_email_string = read_email_and_get_features(file,out)
        if(clean_email_string!="Could not process the file"):
            keywords = word_tokenize(str(clean_email_string))     
            for w in keywords:
                if (w not in stop_words) and (len(w)>2) and ((len(w)<25)):
                    try:
                        all_keywords[w] = all_keywords.get(w,0) + 1
                        keyword_dictonary[w] = keyword_dictonary.get(w,0) + 1
                    except Exception:
                        print ("Issue in gettin the key")
    return keyword_dictonary   
    
stop_words = set(stopwords.words('english'))
file_list = glob.glob(r"E:\Machine learning with Python\SVM for span detection\spam\*") 
keyword_dictonary =  get_keywords(file_list,"spam_out")
sorted_dictonary_keyword = sorted(keyword_dictonary.items(), key=operator.itemgetter(1),reverse=True)
file_list = glob.glob(r"E:\Machine learning with Python\SVM for span detection\non_spam\*") 
keyword_dictonary_1 =  get_keywords(file_list,"non_spam_out")
sorted_dictonary_keyword_1 = sorted(keyword_dictonary_1.items(), key=operator.itemgetter(1),reverse=True)
all_keywords_sorted = sorted(all_keywords.items(), key=operator.itemgetter(1),reverse=True)

#************************* Histogram Analysis Done*****************************

#Histogram_spam_keywords = []
#Histogram_non_spam_keywords = []
#for i in range(0,100):
#    Histogram_spam_keywords.append(sorted_dictonary_keyword[i][0])
#    Histogram_non_spam_keywords.append(sorted_dictonary_keyword_1[i][0])
#words_to_compare = []        
#words_to_compare = list(set(Histogram_non_spam_keywords + Histogram_spam_keywords))
#Histogram_spam_keywords = []
#Histogram_non_spam_keywords = []

#for word in words_to_compare:
#    Histogram_spam_keywords.append(keyword_dictonary.get(word,0))
#    Histogram_non_spam_keywords.append(keyword_dictonary_1.get(word,0))

#*************************Feature Creation ********************************
all_features = []
def create_dataset_for_spam_classification(file_list,output_class_label):
    feature_len = len(vocabulary_dictonary)
    print(feature_len)
    try:
        for file in file_list:
            print(file)
            clean_email_string = read_email_and_get_features(file,"not_required")
            if(clean_email_string!="Could not process the file"):
                feature = [0] * (feature_len + 1) #Defining 1-D array
                feature[allowed_features] = output_class_label #Last col is output class 
                keywords = word_tokenize(str(clean_email_string))
                for word in keywords:
                    try:
                        index = 0
                        index = vocabulary_dictonary.index(word)
                        feature[index] = feature[index] + 1 
                    except Exception:
                        print("")
                all_features.append(feature)     
    except Exception as e:
        print("Issue while processing the email")         

    
def create_csv_dataset(file_path,headers):
    with open(file_path,'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in all_features_processed:
            writer.writerow(row)


vocabulary_dictonary = []
allowed_features = 800
for i in range(0,allowed_features):
    vocabulary_dictonary.append(all_keywords_sorted[i][0])
file_list = glob.glob(r"E:\Machine learning with Python\SVM for span detection\spam\*") 
create_dataset_for_spam_classification(file_list,1)
file_list = glob.glob(r"E:\Machine learning with Python\SVM for span detection\non_spam\*") 
create_dataset_for_spam_classification(file_list,0)
vocabulary_dictonary.append('label')
file_path = os.getcwd() + "\dataset_800.csv"
all_features_processed = []
for arr in all_features:
    proceed = 0
    for i in range(0,allowed_features):
        if(arr[i] > 0):
            proceed = 1
            break
    if(proceed==1):
        all_features_processed.append(arr)
create_csv_dataset(file_path,vocabulary_dictonary)    