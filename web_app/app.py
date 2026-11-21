"""
Flask web app for car price prediction.
Loads the dataset, trains a LinearRegression model, and provides a UI
for users to input car features and receive a predicted price.
"""

import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# ---------- Data loading & model training ----------
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'ford.csv')
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop('price', axis=1)
Y = df['price']

# One‑hot encode categorical columns
categorical_cols = ['transmission', 'fuelType', 'model']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_encoded = X_encoded.astype(int)

# Scale numerical columns
numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

# Train model
model = LinearRegression()
model.fit(X_encoded, Y)

# Store column order for later use
TRAIN_COLUMNS = X_encoded.columns.tolist()

# ---------- Helper to preprocess a single input ----------
def preprocess_input(form_dict):
    """Convert form data (strings) into a DataFrame with the same preprocessing as training."""
    # Build a single‑row DataFrame with appropriate dtypes
    input_df = pd.DataFrame({
        'model': [form_dict.get('model')],
        'year': [int(form_dict.get('year'))],
        'transmission': [form_dict.get('transmission')],
        'mileage': [int(form_dict.get('mileage'))],
        'fuelType': [form_dict.get('fuel_type')],
        'tax': [int(form_dict.get('tax'))],
        'mpg': [float(form_dict.get('mpg'))],
        'engineSize': [float(form_dict.get('engine_size'))],
    })

    # One‑hot encode using same categories as training
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    # Re‑index to match training columns, filling missing columns with 0
    input_encoded = input_encoded.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    # Scale numeric columns
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
    return input_encoded

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Gather form values
        form = request.form
        try:
            processed = preprocess_input(form)
            pred_price = model.predict(processed)[0]
            prediction = f"${pred_price:,.2f}"
        except Exception as e:
            prediction = f"Error: {e}"
    # Populate dropdown choices from training data
    models = sorted(df['model'].unique())
    transmissions = sorted(df['transmission'].unique())
    fuel_types = sorted(df['fuelType'].unique())
    return render_template('index.html',
                           prediction=prediction,
                           models=models,
                           transmissions=transmissions,
                           fuel_types=fuel_types)

if __name__ == '__main__':
    # Run on localhost:5000
    app.run(debug=True)
