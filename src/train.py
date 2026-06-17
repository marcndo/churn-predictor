"""
Responsibility: Everything related to model training.
- Define which model to train and their hyperparameters
- Train each model on processed data
- Evaluate using cross-validation
- Select the best model using f1 score
- Save the best model and preprocessor to disk
"""

import numpy as np
import joblib
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# ============================================
# GET THE DIFFERENT MODELS FOR EXPERIMENTATION
# ============================================

def get_models():
    """Return a dictionary of models to train and compare.
       Return:
       {model_name: model_object}
    """
    models={
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth = 4,
            scale_pos_weight=2.7,
            eval_metric="logloss",
            random_state=42
        )
    }
    return models

# ============================================
# EXPERIMENT WITH DIFFERENT MODELS
# ============================================

def evaluate_with_cv(model, preprocessor, X_train, y_train, cv_folds=5):
    """
    Evaluate model using stratefied k-fold cross-validation.
    Args:
        model: unfitted sklearn-compatible model
        preprocessor: unfitted ColumnTransformer from build_preprocessor
        X_train: feature DataFrame
        y_train: target series
        cv_fold: number of cross validation folds
    """
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42
    )
    start_time = time.time()
    cv_result = cross_validate(
        full_pipeline,
        X_train,
        y_train,
        cv = cv,
        scoring=["f1", "accuracy","roc_auc"],
        return_train_score=False     
    )
    elapsed = time.time() - start_time
    result = {
        "f1_mean": np.mean(cv_result["test_f1"]),
        "f1_std": np.std(cv_result["test_f1"]),
        "accuracy_mean": np.mean(cv_result["test_accuracy"]),
        "roc_auc_mean": np.mean(cv_result["test_roc_auc"]),
        "training_time": elapsed
    }
    return result


# ============================================
# TRAIN ALL MODELS AND COMPARE
# ============================================

def train_and_compare(preprocessor, X_train, y_train):
    """
    Train all models using cross-validation and return
    a comparism of their performance.
    Args:
        preprocessor: unfitted ColumnTransformer
        X_train: feature DataFrame
        y_test: target series
    """
    models = get_models()
    all_results = {}
    print("=" * 60)
    print("MODEL COMPARISON-5-fold stratified Cross-Validation")
    print("Primary metric f1 Score")
    print("=" * 60)
    for name, model in models.items():
        print(f"\nTraining: {name}...")
        results = evaluate_with_cv(
            model, preprocessor, X_train, y_train
        )
        all_results[name] = results
        print(f"F1 Score: {results['f1_mean']:.4f} " 
        f"(±{results['f1_std']:.4f})")
        print(f"   Accuracy: {results['accuracy_mean']:.4f}")
        print(f"   Roc-AUC: {results['roc_auc_mean']}")
        print(f"   Training Time: {results['training_time']}")
    return all_results

# ============================================
# SELECT MODEL
# ============================================

def select_best_model(all_results):
    """
    Select the best model by mean f1 score.
    Args:
        all_result: dict from train_and_compare
    """
    best_name = max(
        all_results,
        key=lambda name: all_results[name]["f1_mean"]
    )
    print("\n" + "=" * 60)
    print("BEST MODEL SELECTED")
    print("=" * 60)
    print(f"Winner:{best_name}")
    print(f"F1 Score: {all_results[best_name]['f1_mean']:.4f}"
    f"(±{all_results[best_name]['f1_std']:.4f})")
    print(f"Reason: Highest mean F1 accross 5 CV folds")
    return best_name

# ============================================
# TRAIN FINAL MODEL ON FULL TRAINING DATA
# ============================================

def train_final_model(best_model_name, preprocessor, X_train, y_train):
    """
    Train the selected best model on the FULL dataset.
    Args:
        best_model_name: string name for best model from select_best_model
        preprocessor: unfitted ColumnTransformer
        X_train: feature DataFrame
        y_train: target series
    """
    models = get_models()
    best_model = models[best_model_name]
    final_pipeline = Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("model", best_model)
    ])
    print(f"\nTraining final {best_model_name} on full training set...")
    final_pipeline.fit(X_train, y_train)
    print("Final model trained")
    return final_pipeline

def save_model(pipeline, filepath="models/best_model.joblib"):
    """
    Save the final fitted pipeline to disk
    Args:
       pipeline: fitted pipeline from train_final_model
       filepath: path to save the trained pipeline
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    file_size = os.path.getsize(filepath)
    print(f"Model save to {filepath} ({file_size:,} bytes)")


