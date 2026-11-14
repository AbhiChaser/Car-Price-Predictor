from json import encoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#EDA
df = pd.read_csv('Car Price Predictor/ford.csv')

X = df.drop(['price'], axis=1)
Y = df['price']

X_one_encoded = pd.get_dummies(X, columns=['transmission', 'fuelType', 'model'], drop_first=True)
X_one_encode = X_one_encoded.astype(int)
print(X_one_encode.head())

from sklearn.preprocessing import LabelEncoder


Xlable = X

X_Lable_Col = ["transmission", "fuelType", "model"]
for col in X_Lable_Col:
    encoder = LabelEncoder()
    Xlable[col] = encoder.fit_transform(Xlable[col])
#print(Xlable.head())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
X_one_encode[numerical_columns] = scaler.fit_transform(X_one_encode[numerical_columns])
#print(X_one_encode.head())
Xlable[X.columns] = scaler.fit_transform(Xlable[X.columns])
#print(Xlable.head())

#Model 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train_X, test_X, train_Y, test_Y = train_test_split(X_one_encode, Y, test_size=0.2, random_state=42)
Model_1 =LinearRegression()
Model_1.fit(train_X, train_Y)

Y_Pred = Model_1.predict(test_X)
print(Y_Pred)
r2 = r2_score(test_Y, Y_Pred)
n = test_X.shape[0]   
p = test_X.shape[1]

adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("R-squared: ", r2 , " Adjusted R-squared: ", adj_r2)


train_X, test_X, train_Y, test_Y = train_test_split(Xlable, Y, test_size=0.2, random_state=42)
Model_2 =LinearRegression()
Model_2.fit(train_X, train_Y)

Y_Pred = Model_2.predict(test_X)
print(Y_Pred)
r2 = r2_score(test_Y, Y_Pred)
   
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("R-squared: ", r2 , " Adjusted R-squared: ", adj_r2)
