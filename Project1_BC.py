# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:08:51 2022

@author: Shariena
"""

###Project 1. Perform classification on breast cancer dataset###
#1. Carry output the proper data preprocessing steps.
#2. Create a feedforward neural network for classification.
#3. Try to achieve a good level of accuracy (>80%)
#Create a git and github repo for this project. Make sure your github is presentable
#Link to dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

#%%
#1. Data preprocessing
#Import modules and load datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"C:\Users\Shariena\Desktop\AI Machine Learning for Industry 4.0\4. DLJ AI05\Projects\1. bc_FFNN\data.csv")
data.head()

#%%
#Counting values of variables in 'diagnosis' and visualize it
data['diagnosis'].value_counts()

plt.figure(figsize=[17,9])
sb.countplot(data['diagnosis'].value_counts())
plt.show()

#%%
#Check for null values

data.isnull().sum()
#feature name ‘Unnamed:32’ contains all the null values so delete or drop that column.

#%%
#droping feature
data.drop(['Unnamed: 32','id'],axis=1,inplace=True)

#%%
#independent variables/Features
X = data.drop('diagnosis',axis=1)
#dependent variables/label
y = data.diagnosis

#%%
#use Scikit learn Label Encoder for encoding the categorical data

#creating the object
le = LabelEncoder()
y = le.fit_transform(y)

#Done with data pre-processing

#%%
# 2. Feedforward NN for Classification problem

#Split data into training and testing

from sklearn.model_selection import train_test_split

SEED = 12345
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

#%%
#Scaling data

#When we create the artificial neural network, 
#we have to scale the data into smaller numbers because 
#the deep learning algorithm multiplies the weights and 
#input data of the nodes and it takes lots of time, 
#So for reducing that time we scale the data

#importing StandardScaler
from sklearn import preprocessing

standardizer = preprocessing.StandardScaler()
X_train = standardizer.fit_transform(X_train)
X_test = standardizer.transform(X_test)

#Data preparation is done

#%%
#start creating the artificial neural network. 
#Import the important libraries that are used for creating ANN

import tensorflow as tf
from keras.layers import Dense

#Create model
model = tf.keras.Sequential()

#layers of NN
#first hidden layer
model.add(Dense(units=9,kernel_initializer='he_uniform',activation='relu',input_dim=30))
#second hidden layer
model.add(Dense(units=9,kernel_initializer='he_uniform',activation='relu'))
# last layer or output layer
model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

#summary of layers
model.summary()

#%%
#model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%%
#Fit ANN into training data
ANN = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=10)
