# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:56:06 2022

@author: damla
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv('Iris.csv')

x=data.iloc[:,1:-1]
y = data.iloc[:,-1:].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

print(x_train.shape)
print(y_train.shape)

knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')

knn.fit(x_train,y_train.ravel())

result = knn.predict(x_test)
print(result)
print(y_test)

# Karmaşıklık matrisi
cm = confusion_matrix(y_test,result)
print(cm)

# Başarı Oranı
accuracy = accuracy_score(y_test, result)
print(accuracy)

print(knn.score(x_test,y_test))

# How to find the optimal k value
score_list=[]
for each in range(1,15):
    knn1=KNeighborsClassifier(n_neighbors=each)
    knn1.fit(x_train,y_train)
    score_list.append(knn1.score(x_test,y_test))
plt.title("EN uygun k değerine göre score")
plt.plot(range(1,15), score_list)
plt.xlabel("k değeri")
plt.ylabel("Score")
plt.show()