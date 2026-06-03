"""
Responsibility: Everything related to model training.
- Defining which models to train
- Fitting models on training data
- Saving trained models to disk.
"""


def get_models():
    """Return a dictionary of models to train and compare."""
    pass


def train_model(model, X_train, y_train):
    """Train a single model and return the fitted model."""
    pass


def save_model(model, filepath):
    """Save a trained model to disk using joblib."""
    pass