# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:48:33 2025

@author: Jiyong Kim
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV File
df = pd.read_csv("C:/Users/김지용/OneDrive - Arizona State University/Desktop/Data project/walmart.csv")

# Convert Date Format
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True) 
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Data Exploration (Sales Distribution)
plt.figure(figsize=(12, 6))
sns.histplot(df['Weekly_Sales'], bins=50, kde=True)
plt.title("Sales Distribution")
plt.show()

# Data Exploration (Monthly Sales Trend)
monthly_sales = df.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='Month', y='Weekly_Sales', hue='Year', data=monthly_sales, marker='o')
plt.title("Monthly Sales Trend (Yearly Comparison)")
plt.xlabel("Month")
plt.ylabel("Total Sales ($)")
plt.xticks(range(1, 13))
plt.legend(title="Year")
plt.grid(True)
plt.show()

# Data Exploration (Holiday vs. Non-Holiday Sales Comparison)
holiday_sales = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
plt.figure(figsize=(8, 5))
sns.barplot(x=holiday_sales.index, y=holiday_sales.values)
plt.xticks([0, 1], ["Non-Holiday", "Holiday"])
plt.title("Average Sales on Holidays vs Non-Holidays")
plt.ylabel("Average Weekly Sales ($)")
plt.show()






