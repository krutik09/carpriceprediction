# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 07:36:50 2021

@author: kolhe
"""


import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df_car = pd.read_csv('car data.csv')
df_car.drop('Car_Name',  axis = 1, inplace=True)

df_car = pd.get_dummies(df_car, drop_first=True)

features = df_car.drop('Selling_Price', axis = 1)
df_target = df_car['Selling_Price']

print(features.columns)

linear_model = LinearRegression()
linear_model.fit(features, df_target)

pickle.dump(linear_model, open('model.pkl', 'wb'))
