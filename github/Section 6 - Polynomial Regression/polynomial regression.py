# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:55:59 2019

@author: shubhendra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("E:/mytraning/datascience/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")
X=data.iloc[:,1:2].values
Y=data.iloc[:,2].values
#Linear Regresssion
from sklearn.linear_model import LinearRegression
reg= LinearRegression()

reg.fit(X,Y)
#polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
reg_1=LinearRegression()
reg_1.fit(X_poly,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,reg.predict(X),color='blue')
plt.title('salary vs expected salary in (linear regression)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()

#polynomial regression
X_grid= np.arange(min(X),max(X),0.1)
X_grid= X_grid.reshape((len(X_grid)),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,reg_1.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('salary vs expected salary in (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()