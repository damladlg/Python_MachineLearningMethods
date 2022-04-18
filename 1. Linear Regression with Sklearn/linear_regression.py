# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 22:57:09 2022

@author: damladlg
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt

data = pd.read_csv('student-mat.csv', sep=';')

# File has 33 attributes. we get the final grade and the attributes that will affect the final grade.
data = data[['G1','G2','G3','studytime','failures','absences','age']]

print(data.head())

sns.pairplot(data)

print(data.dtypes)

Y=np.array(data['G3']) # G3 is the final grade.
X=np.array(data.drop('G3',axis=1))

# Dataset is split into training and testing.
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

# Is taken as an example from the class.
model_linear=LinearRegression()

model_linear.fit(X_train, Y_train) # Model builded.

model_linear.score(X_test,Y_test) # The performance of the model with test data and, the accuracy value is found.
model_linear.score(X_train,Y_train) # With the train data, the model's performance and accuracy value are found.

print('Coefficients: \n', model_linear.coef_)
print('Constant: ', model_linear.intercept_)

new_data =np.array([[10,14,3,0,4,16]])
predict=(model_linear.predict(new_data))
print(predict) # Predicted for final grade with array.

predict=model_linear.predict(X_test)

plt.scatter(Y_test, predict)
print(Y_test)

print(predict)
plt.show()