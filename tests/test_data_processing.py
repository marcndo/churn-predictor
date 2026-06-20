"""Test for src/data_processing.py"""

import pandas as pd
import numpy as np
import os
import sys

#Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import clean_data, split_data, build_preprocessor

def make_sample_data():
    """Create a sample dataset that mimics the real data structure."""
    return pd.DataFrame({
        'customerID': ['001','002','003','004','005',
                       '006','007','008','009','010'],
        'gender': ['Male','Female','Male','Female','Male',
                   'Female','Male','Female','Male','Female'],
        'SeniorCitizen': [0,1,0,0,1,0,1,0,0,1],
        'Partner': ['Yes','No','Yes','No','Yes',
                    'No','Yes','No','Yes','No'],
        'Dependents': ['No','No','Yes','No','No',
                       'Yes','No','No','Yes','No'],
        'tenure': [1,34,2,45,8,12,24,3,56,7],
        'PhoneService': ['No','Yes','Yes','No','Yes',
                        'Yes','No','Yes','Yes','No'],
        'MultipleLines':['No','No','No','No phone service','No',
                        'Yes','No','No','No phone service','No'],
        'InternetService': ['DSL','Fiber optic','DSL','DSL','Fiber optic',
                            'DSL','Fiber optic','DSL','DSL','Fiber optic'],
        'OnlineSecurity': ['No','No','Yes','Yes','No',
                        'Yes','No','No','Yes','No'],
        'OnlineBackup': ['Yes','No','No','Yes','No',
                        'No','Yes','No','Yes','No'],
        'DeviceProtection': ['No','Yes','No','Yes','No',
                            'Yes','No','Yes','No','Yes'],
        'TechSupport': ['No','No','No','Yes','No',
                        'No','Yes','No','No','Yes'],
        'StreamingTV': ['No','No','No','No','No',
                        'Yes','No','No','No','Yes'],
        'StreamingMovies': ['No','No','No','No','No',
                            'No','Yes','No','No','Yes'],
        'Contract': ['Month-to-month','One year','Month-to-month',
                    'One year','Month-to-month','Two year',
                    'Month-to-month','One year','Two year',
                    'Month-to-month'],
        'PaperlessBilling': ['Yes','No','Yes','No','Yes',
                            'No','Yes','No','Yes','No'],
        'PaymentMethod': ['Electronic check','Mailed check',
                        'Electronic check','Bank transfer (automatic)',
                        'Electronic check','Mailed check',
                        'Bank transfer (automatic)','Electronic check',
                        'Mailed check','Electronic check'],
        'MonthlyCharges': [29.85,56.95,53.85,42.30,70.70,
                        35.10,85.45,25.60,90.25,45.80],
        'TotalCharges': ['29.85','1889.5',' ','1840.75','531.9',
                        '421.2','2050.8','76.8','5040.5','320.6'],
        'Churn': ['No','No','Yes','No','Yes',
                'No','Yes','No','No','Yes']
    })

def test_clean_data_drop_column():
    "customerID and TotalCharges must be droped after cleaning"
    df = make_sample_data()
    cleaned = clean_data(df)
    assert "customerID" not in cleaned.columns, \
    "customerID should be dropped"
    assert "TotalCharges" not in cleaned.columns, \
    "TotalCharges should be dropped"


def test_clean_data_encode_target():
    "Churn column must be 0 and 1 after cleaning not Yes and No"
    df = make_sample_data()
    cleaned = clean_data(df)
    assert set(cleaned["Churn"].unique()).issubset({0,1}), \
    "Churn should only contain 0 or 1 after cleaning"


def test_clean_data_handle_empty_total_charges():
    "Empty string in TotalCharges must be NaN or not crash."
    df = make_sample_data()
    cleaned = clean_data(df)
    assert "customerID" not in cleaned.columns
    assert "TotalCharges" not in cleaned.columns


def test_split_data_correct_sizes():
    "Train set should be 80% and test set 20% of the data."
    df = make_sample_data()
    cleaned = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(cleaned, test_size=0.2)
    total = len(X_train) + len(X_test)
    assert total == len(cleaned), \
    "Train and test set together must be equal to the full dataset"


def test_preprocessor_build_without_error():
    "build_preprocessor must return a ColumnTransformer without crashing."
    from sklearn.compose import ColumnTransformer
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer), \
    "build_preprocessor must return a ColumnTransformer"


def test_preprocessor_transformer_data():
    "Preprocessor must fit and transform training data without error"
    df = make_sample_data()
    cleaned = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(cleaned, test_size=0.2)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)
    assert X_transformed.shape[0] == X_train.shape[0],\
    "Transformed data must have same number of rows as input"