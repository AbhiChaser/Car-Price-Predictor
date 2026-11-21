
# Ford Car Price Prediction

## Overview
This project predicts the selling price of Ford cars using Machine Learning.
It uses Linear Regression to learn relationships between car specifications
and their market prices.

## Machine Learning Model
- Algorithm: Linear Regression
- Learning Type: Supervised Learning
- Problem Type: Regression

## Dataset
Dataset: ford.csv

### Target Variable
- price

### Features
- year
- transmission
- mileage
- fuelType
- tax
- mpg
- engineSize
- model

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Workflow
1. Load the Ford dataset
2. Separate features and target
3. Encode categorical variables using One-Hot Encoding
4. Split data into training and testing sets
5. Scale numerical features using StandardScaler
6. Train the Linear Regression model
7. Predict car prices
8. Evaluate model performance

## Data Preprocessing

### One-Hot Encoding
One-Hot Encoding converts categorical variables into binary columns.

Used for:
- transmission
- fuelType
- model

Implemented using `pd.get_dummies()` with `drop_first=True`.

### Feature Scaling
StandardScaler is used to standardize numerical features:
- year
- mileage
- tax
- mpg
- engineSize

## Model Training
Linear Regression is trained using 80% of the dataset.
The remaining 20% is used for testing.

## Evaluation Metrics

- R² Score — measures explained variance
- Adjusted R² — accounts for number of predictors
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Structure

```text
Car Price Predictor/
│
├── ford.csv
├── car_price_prediction.py
└── README.md
```

## Run

```bash
python car_price_prediction.py
```

## Objective
To build a regression model capable of predicting Ford car prices
based on their specifications and evaluate its predictive performance.

## Future Improvements
- Try Random Forest Regression
- Try Decision Tree Regression
- Compare multiple ML algorithms
- Improve model accuracy through cross-validation
- Develop a web-based car price prediction application
