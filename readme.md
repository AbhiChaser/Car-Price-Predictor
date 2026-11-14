# 🚗 Ford Car Price Predictor

A Machine Learning project focused on predicting the market prices of Ford cars using historical vehicle data. The project explores data analysis, visualization, categorical encoding, feature scaling, and Linear Regression to understand the factors influencing car prices.

## 📌 Project Overview

This project uses the Ford car dataset (`ford.csv`) to explore relationships between vehicle characteristics and their selling prices.

It implements two different data preprocessing approaches and evaluates their performance using a Linear Regression model.

## ✨ Features

- Exploratory Data Analysis (EDA)
- Statistical analysis and dataset inspection
- Missing-value analysis
- Price distribution visualization
- Correlation heatmap
- Price analysis across transmission types, fuel types, and car models
- One-Hot Encoding for categorical variables
- Label Encoding for categorical variables
- StandardScaler for numerical feature scaling
- Linear Regression model training
- Model evaluation using R² and Adjusted R²

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing and machine learning |

## 📊 Dataset Features

The dataset contains information about Ford vehicles, including:

- `model` – Ford car model
- `year` – Manufacturing year
- `price` – Vehicle price (target variable)
- `transmission` – Transmission type
- `mileage` – Distance driven
- `fuelType` – Fuel category
- `tax` – Vehicle tax
- `mpg` – Miles per gallon
- `engineSize` – Engine capacity

## ⚙️ Machine Learning Workflow

### 1. Exploratory Data Analysis

The dataset is analyzed using Pandas and visualized using Matplotlib and Seaborn.

Visualizations include:
- Price distribution histogram
- Correlation heatmap
- Transmission vs. price boxplot
- Fuel type vs. price boxplot
- Car model vs. price boxplot

### 2. Data Preprocessing

Two encoding strategies are explored:

**Approach A: One-Hot Encoding**
- Converts categorical variables into binary columns.
- Uses `drop_first=True` to reduce redundant features.

**Approach B: Label Encoding**
- Converts categorical values into numerical labels.
- Applies StandardScaler to the resulting features.

Numerical features are standardized using StandardScaler.

### 3. Model Training

A Linear Regression model is trained using both preprocessing approaches.

The dataset is split into:
- 80% Training data
- 20% Testing data

Random state: `42`

### 4. Model Evaluation

The following metrics are used:

- **R² Score:** Measures how well the model explains price variation.
- **Adjusted R² Score:** Accounts for the number of predictors used in the model.

The two encoding approaches can then be compared to understand their impact on model performance.

## 🚀 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ford-car-price-predictor.git
```

Navigate into the project:

```bash
cd ford-car-price-predictor
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Ensure `ford.csv` is available at the expected dataset path, then run your Python script:

```bash
python main.py
```

## 📈 Future Improvements

- Experiment with Decision Tree and Random Forest Regressors.
- Implement cross-validation for more reliable model evaluation.
- Perform hyperparameter tuning.
- Compare additional regression algorithms.
- Improve categorical feature encoding.
- Build an interactive car price prediction interface.
- Deploy the trained model using Flask or FastAPI.

## 👨‍💻 Author

**Abhinav Raj**

Machine Learning | Deep Learning | Python | Game Development

*This project is currently under development. Model optimization, improved preprocessing, and additional regression algorithms are planned for future iterations.*