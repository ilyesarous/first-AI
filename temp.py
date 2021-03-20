# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('weight-height.csv')

#giving the x and y values from the dataset
x = dataset.iloc[:,1].values.reshape(-1, 1)
y = dataset.iloc[:,2].values.reshape(-1, 1)

from sklearn.model_selection import train_test_split

xTrain,xTest,yTrain,yTest = train_test_split(x,y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(xTrain,yTrain)
#the main ML command
yPred = regressor.predict(xTest)

#drowing the histogram
plt.scatter(xTrain, yTrain, color='red')
plt.plot(xTrain, regressor.predict(xTrain),color ='blue')
plt.xlabel('high')
plt.ylabel('weigth')
plt.show()

#an example to see if its working
yPred2 = regressor.predict([[10]])
print(yPred2)
print('-------------------')
print('Train: ',regressor.score(xTrain, yTrain))
print('Test: ',regressor.score(xTest, yTest))
