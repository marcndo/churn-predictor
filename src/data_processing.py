"""
Responsibility: Everything that happens to data before it reaches the model.
- Loading raw data
- Cleaning (fixing types, handling missing values)
- Feature engineering
- Splitting into train/test sets
- Encoding and scaling.
"""


def load_data(filepath):
    """Load raw CSV data and return a DataFrame."""
    pass


def clean_data(df):
    """Fix data types, handle missing values, drop irrelevant columns."""
    pass


def engineer_features(df):
    """Create new features from existing ones based on EDA findings."""
    pass


def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    pass


def build_preprocessor(numerical_cols, categorical_cols):
    """Build and return a scikit-learn preprocessing pipeline."""
    pass