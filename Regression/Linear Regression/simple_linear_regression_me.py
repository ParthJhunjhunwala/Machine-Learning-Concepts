# -*- coding: utf-8 -*-
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting dataset into Test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting the linear regression model to teh training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_prediciton = regressor.predict(X_test)

#Visualizing the training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experince (Training Set)')
plt.xlabel('Years of Experince')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experince (Test Set)')
plt.xlabel('Years of Experince')
plt.ylabel('Salary')
plt.show()