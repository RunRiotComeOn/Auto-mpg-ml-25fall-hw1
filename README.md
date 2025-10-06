# ECS 171 Homework 1: Auto MPG Analysis

This repository contains the implementation and analysis for ECS 171 Homework Set 1, focusing on predicting and classifying automobile fuel efficiency (MPG) using the Auto MPG dataset from the UCI Machine Learning Repository.

## Dataset

The Auto MPG dataset contains 398 cars with 9 features. After removing 6 records with missing values, we work with 392 samples.

**Features:**
- cylinders
- displacement
- horsepower
- weight
- acceleration
- model year
- origin
- car name (excluded from analysis)

**Target Variable:** MPG (miles per gallon)

## Project Overview

This homework implements machine learning algorithms from scratch to analyze the relationship between car attributes and fuel efficiency. The project covers:

1. **MPG Categorization** - Dividing cars into 4 equally-sized bins (Low, Medium, High, Very High)
2. **Feature Analysis** - Creating scatterplot matrices to identify informative feature pairs
3. **Polynomial Regression (Single Variable)** - Custom implementation using Ordinary Least Squares
4. **Performance Evaluation** - Testing polynomial orders 0-3 on individual features
5. **Multivariate Polynomial Regression** - Extending to handle all 7 features simultaneously
6. **Logistic Regression Classification** - Classifying cars into MPG categories
7. **Feature Normalization** - Comparing performance with min-max scaling
8. **Prediction** - Forecasting MPG for hypothetical car specifications

## File Structure

```
.
├── Q1_Q2.py                # Data cleaning, asigning bins to mpg values and using variance (Question 1-2)
├── Q3_Q4.py                # Single-variable polynomial regression (Questions 3-4)
├── Q5.py                   # Multivariate polynomial regression (Question 5)
├── Q6_Q7.py               # Logistic regression classification (Questions 6-7)
├── Q8.py                  # MPG prediction for new car (Question 8)
├── auto-mpg/
│   └── auto-mpg-clean.csv # Cleaned dataset
├── plots/                 # Generated visualization plots
└── ckpts/                 # Saved models and scalers
```

## Implementation Details

### Questions 1 & 2: MPG Categorization and Feature Analysis (`Q1_Q2.ipynb`)

This Jupyter notebook combines the analysis for Questions 1 and 2, performing MPG categorization and comprehensive feature pair analysis.


### Single-Variable Polynomial Regression (`Q3_Q4.py`)

**Class: `SinglePolyRegression`**
- Custom implementation using OLS estimator
- Supports polynomial degrees 0-3
- Creates polynomial features: [1, x, x², x³, ...]
- Methods: `fit()`, `predict()`, `mse()`

**Key Results:**
- Train/test split: First 292 samples for training, remaining 100 for testing
- Evaluates each feature individually across polynomial orders 0-3
- Generates 7 plots showing test set predictions with all polynomial fits
- Identifies best polynomial degree and most informative feature

### Multivariate Polynomial Regression (`Q5.py`)

**Class: `MultiPolyRegression`**
- Extends polynomial regression to handle multiple features
- Second-order implementation includes:
  - Constant term (1)
  - Linear terms (x₁, x₂, ..., x₇)
  - Quadratic terms (x₁², x₂², ..., x₇²)
  - Total: 15 terms for degree 2

**Training Configurations:**
- Degree 0: Constant model (baseline)
- Degree 1: Linear model with 7 features
- Degree 2: Quadratic model with 15 terms

### Logistic Regression Classification (`Q6_Q7.py`)

**MPG Categories:**
- Low: MPG ≤ 17.00
- Medium: 17.00 < MPG ≤ 22.75
- High: 22.75 < MPG ≤ 29.00
- Very High: MPG > 29.00

**Implementation:**
- Uses scikit-learn's LogisticRegression
- Question 6: Classification without normalization
- Question 7: Classification with MinMaxScaler normalization
- Metrics: Macro-averaged precision, classification report, confusion matrix

### Prediction Application (`Q8.py`)

Predicts MPG and category for a hypothetical 1981 USA car with specifications:
- 4 cylinders
- 400 cc displacement
- 150 horsepower
- 3500 lb weight
- 8 m/sec² acceleration
- Model year: 81
- Origin: 1 (USA)

Uses both:
1. Second-order polynomial regression → MPG value → Category
2. Logistic regression → Direct category prediction with probabilities

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
pickle
```

## Usage

### Run Individual Questions

```bash
# Questions 3-4: Single-variable polynomial regression
python Q3_Q4.py

# Question 5: Multivariate polynomial regression
python Q5.py

# Questions 6-7: Logistic regression classification
python Q6_Q7.py

# Question 8: Prediction for new car
python Q8.py
```

### Expected Outputs

**Q3_Q4.py:**
- Training MSE table for all features and polynomial degrees
- Testing MSE table for all features and polynomial degrees
- 7 plots saved in `./plots/` directory
- Best performing polynomial degree and feature

**Q5.py:**
- Training MSE for degrees 0, 1, 2
- Testing MSE for degrees 0, 1, 2
- Best performing degree
- Saved models in `./ckpts/` directory

**Q6_Q7.py:**
- Training and testing precision (without normalization)
- Training and testing precision (with normalization)
- Classification reports and confusion matrices
- Saved models and scaler

**Q8.py:**
- Predicted MPG value
- MPG category from polynomial regression
- MPG category from logistic regression
- Category probabilities

## Key Findings

The implementation demonstrates:
- Trade-offs between model complexity and generalization
- Impact of feature selection on prediction accuracy
- Benefits and limitations of feature normalization
- Comparison between regression and classification approaches

