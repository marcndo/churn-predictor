# Customer Churn Prediction API

## Problem
A telecom company loses revenue every time a customer cancels their subscription. This project builds a machine learning model that predicts 
which customers are likely to churn, enabling the business to intervene proactively.

## Approach
- Exploratory data analysis to understand churn patterns
- Preprocessing pipeline: encoding, scaling, missing value handling
- Model comparison: Logistic Regression vs Random Forest vs XGBoost
- Best model served via FastAPI REST endpoint

## Tech stack
Python · scikit-learn · XGBoost · FastAPI · Docker

## Results

Best model: **XGBoost** (selected via 5-fold stratified cross-validation, F1 score)

| Metric | Score |
|--------|-------|
| F1 Score | 0.632 |
| Precision | 0.522 |
| Recall | 0.789 |
| Accuracy | 0.758 |
| ROC-AUC | 0.845 |

**Why XGBoost was selected:** Despite having lower raw accuracy than 
Logistic Regression, XGBoost achieved the highest F1 score — the right 
metric for this imbalanced dataset (73% non-churn / 27% churn). A model 
that simply predicted "no churn" for every customer would achieve ~73% 
accuracy while being completely useless to the business.

**Business interpretation:** The model catches 79% of customers who will 
actually churn, enabling proactive retention offers. The top predictive 
signal is contract type — month-to-month customers churn at dramatically 
higher rates than annual contract customers, confirming patterns 
discovered during exploratory data analysis.

[Confusion Matrix](data/plots/confusion_matrix.png)
[Feature Importance](data/plots/feature_importance.png)

## How to run
_To be filled after deployment_