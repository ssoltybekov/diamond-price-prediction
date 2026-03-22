# 💎 Diamond Price Prediction (Linear Models)

## 📌 Overview

This project focuses on predicting diamond prices using classical machine learning models.
The goal is to explore how numerical and categorical features influence price and evaluate different linear models.

Dataset includes features such as:

* `carat` — weight of the diamond
* `depth`, `table` — physical properties
* `cut`, `color`, `clarity` — categorical quality indicators

---

## ⚙️ Data Preprocessing

### 🔹 Filtering

* Removed outliers by keeping only diamonds with:

```python
carat < 2.5
```

### 🔹 Feature Selection

Used the following features:

* Numerical: `carat`, `depth`, `table`
* Categorical: `cut`, `color`, `clarity`
* Target: `price`

### 🔹 Encoding

Categorical variables were converted using One-Hot Encoding:

```python
pd.get_dummies(..., drop_first=True)
```

* Avoids multicollinearity (dummy variable trap)

### 🔹 Scaling

Features were standardized using:

```python
StandardScaler()
```

---

## 📊 Exploratory Data Analysis

### Scatter Plot

* Visualized relationship between `carat` and `price`
* Strong positive correlation observed

### Correlation Matrix

* Used heatmap to analyze feature relationships
* No extreme correlations detected

---

## 📉 Model Training

Split:

```python
train_test_split(test_size=0.2, random_state=42)
```

Models used:

* Linear Regression
* Ridge Regression
* Lasso Regression

---

## 📏 Evaluation Metrics

* R² Score
* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)

---

## 📊 Results

| Model             | R²    | RMSE   | MAE    |
| ----------------- | ----- | ------ | ------ |
| Linear Regression | ~0.92 | Low    | Low    |
| Ridge             | ~0.92 | Low    | Low    |
| Lasso             | Lower | Higher | Higher |

---

## 🔍 Multicollinearity Check

* Used Variance Inflation Factor (VIF)
* All VIF values < 10
  ✅ No multicollinearity issues detected

---

## 📉 Residual Analysis

* Residuals plotted against predictions
* Distribution visualized with histogram

Findings:

* Residuals centered around 0
* Slight non-linearity may exist
* Model assumptions mostly satisfied

---

## 🔁 Cross-Validation

Used 5-Fold Cross Validation:

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

Target transformation:

```python
y_log = np.log(price)
```

Results:

* Very stable performance (±0.0004)
* Consistent across folds

---

## 🧠 Key Insights

1. Linear Regression and Ridge perform almost identically
2. Lasso reduces performance slightly but performs feature selection
3. Model is highly stable across different splits
4. `carat` is the strongest predictor of price
5. Log transformation improves model stability

---

## 🚀 Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib / Seaborn
* Scikit-learn
* Statsmodels

---

## 📌 Conclusion

Linear models perform surprisingly well on this dataset.
Regularization does not significantly improve performance, indicating:

* Low multicollinearity
* Strong linear relationships

Future improvements:

* Try non-linear models (Random Forest, XGBoost)
* Feature engineering
* Hyperparameter tuning
