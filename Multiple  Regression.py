# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:30:17 2020

@author: Mutunga
"""

#MULTIPLE REGRESSION 

import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

#Importing the dataset

dataset = pd.read_csv('insurance.csv')
dataset.head()

#Creating the Matrix of Features and Target variable vector
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
y = y.reshape(len(y),1)
print(X)
print(y)

# Encoding Categorical variables "sex" and "smoker"
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder() #() Everything we are not specifying
X[:,1]=le.fit_transform(X[:,1]) 
print(X[:,1])
X[:,4]=le.fit_transform(X[:,4]) 
print(X[:,4])

#OneHotEncoding of the variable "region"
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[5])], remainder='passthrough') 
X=np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Training the Multiple Regression algorthim on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Evaluation Metric
from sklearn import metrics
from sklearn.metrics import r2_score
r_square= np.round(r2_score(y_test, y_pred)*100,1)
print('R Squared Score:',r_square,'%')
