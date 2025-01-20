# 🛒 Retail Sales Prediction & Analysis using Machine Learning

## 📌 Overview
This project analyzes Walmart's retail sales data and builds a **machine learning model** to predict weekly sales.  
Using Python (`pandas`, `seaborn`, `scikit-learn`), we perform **data exploration, visualization, and forecasting**.  

✅ **Key Features:**  
- **Exploratory Data Analysis (EDA):** Sales trends, seasonal patterns, holiday impact (`retail_data_exploration.py`)  
- **Data Visualization:** Histograms, line charts, and bar plots using `matplotlib` & `seaborn`  
- **Machine Learning Model:** `RandomForestRegressor` to predict weekly sales (`retail_sales_prediction.py`)  
- **Hyperparameter Tuning:** `RandomizedSearchCV` for optimal model performance  

---

## 📂 Dataset Information
- **Filename:** `walmart.csv`
- **Source:** Walmart sales data (fictional dataset)
- **Columns:**
  - `Store` - Store ID  
  - `Date` - Sales date  
  - `Weekly_Sales` - Weekly sales revenue  
  - `Holiday_Flag` - Holiday indicator (1 = holiday week, 0 = non-holiday week)  
  - `Temperature` - Temperature in the region  
  - `Fuel_Price` - Fuel price in the region  
  - `CPI` - Consumer Price Index  
  - `Unemployment` - Unemployment rate  

---

## 🛠️ How to Run the Project
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/retail-sales-prediction.git
cd retail-sales-prediction
```

### **2️⃣ Install Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **3️⃣ Run the Exploratory Data Analysis (EDA) Script**
```bash
python retail_data_exploration.py
```
✅ **This script will generate visualizations and insights about sales trends.**  

### **4️⃣ Run the Machine Learning Model**
```bash
python retail_sales_prediction.py
```
✅ **This script will train the model, predict sales, and evaluate performance.**  

---

## 🔍 Scripts Overview
### **`retail_data_exploration.py` (EDA & Visualization)**
This script performs **exploratory data analysis (EDA)** and generates key visualizations:  
- **Sales Distribution Analysis**  
- **Monthly Sales Trend Analysis**  
- **Holiday vs Non-Holiday Sales Comparison**  

### **`retail_sales_prediction.py` (Machine Learning)**
This script trains a **RandomForestRegressor model** for sales forecasting, including:  
- **Feature Engineering (`Previous_Week_Sales`, `Month`, `Week`, etc.)**  
- **Data Preprocessing (Outlier removal, Log transformation, Scaling)**  
- **Hyperparameter Tuning (`RandomizedSearchCV`)**  
- **Model Performance Evaluation (`MAE`, `MSE`)**  

---

## 🔥 Machine Learning Model & Performance
### **Model:** Random Forest Regressor (`RandomForestRegressor`)  
- **Feature Engineering:** `Previous_Week_Sales`, `Month`, `Week`, `CPI`, etc.  
- **Data Preprocessing:** **Outlier removal, log transformation, feature scaling**  
- **Hyperparameter Optimization:** **`RandomizedSearchCV`**  

### **Performance Metrics**
| Metric | Value |
|--------|-------|
| **Mean Absolute Error (MAE)** | **0.06** |
| **Mean Squared Error (MSE)**  | **0.01** |

📈 **The model achieved high accuracy, significantly improving predictions!** 🚀  

---

## 📌 Project Structure
```
retail-sales-prediction/
│── walmart.csv                     # Sales dataset
│── retail_data_exploration.py       # Python script (EDA & Visualization)
│── retail_sales_prediction.py       # Python script (Machine Learning Model)
│── README.md                        # Project documentation
```

---

## 📢 Contact & Contributions
📧 **Author:** Jiyong Kim  
💼 **LinkedIn:**(https://www.linkedin.com/in/jiyong-kim-98368823a/)    

💡 **Feel free to contribute!** If you have suggestions or improvements, submit a pull request! 🚀  
