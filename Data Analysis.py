from json import encoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Car Price Predictor/ford.csv')
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

sns.histplot(data=df, x='price', bins=30, kde=True)
plt.show()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()
sns.boxplot(data=df, x='transmission', y='price')
plt.show()
sns.boxplot(data=df, x='fuelType', y='price')
plt.show()
sns.boxplot(data=df, x='model', y='price')
plt.show()