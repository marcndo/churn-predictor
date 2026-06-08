"""
Responsibility: Everything that transforms raw data into model ready data.
Every decision here traces back to the findings in notebooks/01_eda.ipynb

Auther: Marce Ndowah
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

# Constants

# EDA finding: CustomerID is a unique identifier, not a predictor
# EDA finding: TotalCharges is redundant with tenure * MonthlyCharges

COLUMNS_TO_DROP = ['customerID', 'TotalCharges']

# EDA finding: These columns contains Yes/No
# They are encoded differently from mult-category columns

BINARY_COLUMNS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling","Churn"
]

# EDA finding: these columns have multiple categories
# OneHotEncoding creates one binary columns per category

CATEGORICAL_COLUMNS = [
    "InternetService", "Contract", "PaymentMethod",
    "MultipleLines","OnlineSecurity","OnlineBackup",
    "DeviceProtection", "TechSupport","StreamingTV","StreamingMovies"
]

# EDA finding: these columns are continuous numerical values
# StandardScaler normalizes their range

NUMERICAL_COLUMNS = [
    "tenure", "MonthlyCharges"
]

# Target column: What we want to predict
TARGET_COLUMN = "Churn"

#=================================
# STEP 1 - LOAD DATA
#==================================

def load_data(filepath):
    """
    Load raw CSV data from disk and return a DataFrame.
    args:
       filepath: path to the CSV file
    Returns:
        pandas dataframe with raw data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found at: {filepath}\n"
            f"Download it using the wget command in your notebook"
        )
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


#==========================================
# STEP 2 - CLEAN DATA
#==========================================

def clean_data(df):
    """
    Fix data quality issues discovered during EDA.
    Args:
        df: raw data from load_data()
    Returns:
        clean DataFrame
    """
    df = df.copy()
    # TotalCharges: Replace empty strings with NaN, convert to float
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop columns with no predictive value
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # Encode Yes/No columns
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    assert df["SeniorCitizen"].isin([0, 1]).all(), \
    "SeniorCitizen should only contain 0 or 1"
    print(f"Data cleaned: {df.shape[0]} row, {df.shape[1]} columns")
    print(f"Missing values after cleaned\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


#==========================================
# STEP 3 - SPLIT DATA
#==========================================
def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    Args:
        df: cleaned DataFrame from clean_data()
        test_size: fraction of dataset reserver for testing
        random_state: random seed for reproducibility
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = test_size,
        random_state = random_state,
        stratify = y
        )
    print(f"Train set: {X_train.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows")
    print(f"Churn rate in train set: {y_train.mean():.2%}")
    print(f"Churn rate in test set: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


#==========================================
# STEP 4 - BUILD PREPROCESSOR
#==========================================
def build_preprocessor():
    """Build and return a scikit-learn ColumnTransformer
    that applies the correct transformation to each columns
    type.
    Returns:
       fitted-ready ColumnTransformer preprocessor
    """
    # pipeline for numerical column
    numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])

    # pipeline for categorical columns
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # combine pipeline
    preprocessor = ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, NUMERICAL_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS)
    ])
    return preprocessor

#==========================================
# STEP 5 - SAVE AND LOAD PREPROCESSOR
#==========================================

def save_processor(preprocessor, filepath="models/preprocessor.joblib"):
    """Save fitted processor to disk """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)
    print(f"Preprocessor saved to {filepath}")

def load_preprocessor(filepath="models/preprocessor.joblib"):
    """Load saved preprocessor from disk"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processor not found at {filepath}")
    return joblib.load(filepath)
