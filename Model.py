
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# DATASET
# ==========================================

df = pd.read_csv('Car Price Predictor/ford.csv')

X = df.drop('price', axis=1)
Y = df['price']


# ==========================================
# ONE-HOT ENCODING
# ==========================================

X_encoded = pd.get_dummies(
    X,
    columns=['transmission', 'fuelType', 'model'],
    drop_first=True
)

X_encoded = X_encoded.astype(int)


# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

numerical_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

X_encoded[numerical_columns] = scaler.fit_transform(
    X_encoded[numerical_columns]
)


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

train_X, test_X, train_Y, test_Y = train_test_split(
    X_encoded,
    Y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# MODEL TRAINING
# ==========================================

Model = LinearRegression()

Model.fit(train_X, train_Y)


# ==========================================
# PREDICTION
# ==========================================

Y_Pred = Model.predict(test_X)

print("\nPredicted Prices:")
print(Y_Pred)


# ==========================================
# EVALUATION
# ==========================================

r2 = r2_score(test_Y, Y_Pred)

n = test_X.shape[0]
p = test_X.shape[1]

adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

mse = mean_squared_error(test_Y, Y_Pred)
rmse = np.sqrt(mse)


print("\nModel Performance")
print("-------------------------")
print("R-squared:", r2)
print("Adjusted R-squared:", adj_r2)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
