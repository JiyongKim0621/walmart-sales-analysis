# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:52:09 2025

@author: Jiyong Kim
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load CSV File
df = pd.read_csv("C:/Users/김지용/OneDrive - Arizona State University/Desktop/Data project/walmart.csv")

# Convert Date Format (Auto Detection)
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)

# Add Year, Month, and Week Columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week

# Add Previous Week Sales (Moving Average by Store)
df = df.sort_values(by=['Store', 'Date'])
df['Previous_Week_Sales'] = df.groupby('Store')['Weekly_Sales'].shift(1)
df['Previous_Week_Sales'] = df['Previous_Week_Sales'].fillna(df['Weekly_Sales'].mean())

# Remove Outliers (Eliminate Extreme Values in Sales Data)
q1 = df['Weekly_Sales'].quantile(0.01)
q3 = df['Weekly_Sales'].quantile(0.99)
df = df[(df['Weekly_Sales'] >= q1) & (df['Weekly_Sales'] <= q3)]

# Log Transformation (Adjust Sales Data Distribution)
df['Weekly_Sales'] = np.log1p(df['Weekly_Sales'])
df['Previous_Week_Sales'] = np.log1p(df['Previous_Week_Sales'])

# Data Normalization (Temperature, Fuel_Price, CPI, Unemployment)
scaler = StandardScaler()
df[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']] = scaler.fit_transform(df[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']])

# Define Training Data
features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Month', 'Week', 'Previous_Week_Sales']
target = 'Weekly_Sales'

X = df[features]
y = df[target]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(50, 300, 50),
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42), 
    param_distributions=param_dist, 
    n_iter=10,  
    cv=3, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1,  # Enable Parallel Processing
    random_state=42
)

# Train Model
random_search.fit(X_train, y_train)

# Find Best Model
best_model = random_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate Model Performance
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)

print(f"✅ MAE: {mae_best:.2f}")
print(f"✅  MSE: {mse_best:.2f}")

# Add Visualization: Actual vs Predicted Comparison
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Actual vs Predicted Sales (Scatter Plot)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_best, alpha=0.5, color='blue')  # Actual vs Predicted Values
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal Prediction Line (y=x)
plt.xlabel("Actual Weekly Sales")
plt.ylabel("Predicted Weekly Sales")
plt.title("Actual vs Predicted Weekly Sales")
plt.show()

# Prediction Error Histogram (Assess Model Accuracy)
error = y_test - y_pred_best  # Actual - Predicted (Error Calculation)
plt.figure(figsize=(10, 5))
sns.histplot(error, bins=50, kde=True, color='blue')
plt.axvline(x=0, color='red', linestyle='--')  # Central Reference Line
plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()


# Total Sales by Store (Additional Analysis)
plt.figure(figsize=(15, 6))
store_sales = df.groupby('Store')['Weekly_Sales'].sum().reset_index()
sns.barplot(x='Store', y='Weekly_Sales', data=store_sales)
plt.title("Total Sales by Store")
plt.xlabel("Store Number")
plt.ylabel("Total Sales ($)")
plt.xticks(rotation=90)
plt.show()

