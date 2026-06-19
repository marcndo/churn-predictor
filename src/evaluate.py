"""
Final evaluation of the trained model on 
held-out test set.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import joblib
import os
from sklearn.metrics import(
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score
)

#================================
#EVALUATE MODEL ON TEST SET
#================================

def evaluate_model(pipeline, X_test, y_test,model_name="XGBoost"):
    """
    Evaluate the fitted pipeline on the held-out set.
    Args:
        pipeline: fitted pipeline from the train_final_model
        X_test: raw feature DataFrame(uncleaned)
        y_test: true label series
    Return:
        dic of metric name -> metric value
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    # compute metrics
    metrics = {
    "model":     model_name,
    "accuracy":  accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall":    recall_score(y_test, y_pred, zero_division=0),
    "f1":        f1_score(y_test, y_pred, zero_division=0),
    "roc_auc":   roc_auc_score(y_test, y_prob)
    }

    return metrics, y_pred, y_prob


#================================
#PRINT PERFORMANCE REPORT
#================================
def print_report(metrics, y_test, y_pred):
    """
    Print a formatted performance report to console.

    Args:
        metrics: dict from evaluate_model()
        y_test: true labels
        y_pred: predicted labels
    """
    print("\n" + "=" * 60)
    print(f"FINAL TEST SET EVALUATION — {metrics['model']}")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nDetailed Classification Report:")
    print("-" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=["No Churn", "Churn"]
    ))

#================================
#CONFUSION MATRIX PLOT
#================================

def plot_confusion_matrix(y_test, y_pred, model_name="XGBoost",
                        save_path="data/confusion_matrix.png"):
    """
    Plot and save the confusion matrix.
    Args:
    y_test: true labels
    y_pred: predicted labels
    save_path: where to save the plot
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        linewidths=0.5
    )
    plt.title(f'Confusion Matrix — {model_name}', fontsize=13)
    plt.ylabel('Actual', fontsize=11)
    plt.xlabel('Predicted', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

    # Print interpretation
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix breakdown:")
    print(f"  True Negatives  (correctly predicted no churn): {tn}")
    print(f"  False Positives (false alarms):                 {fp}")
    print(f"  False Negatives (missed churners):              {fn}")
    print(f"  True Positives  (correctly caught churners):    {tp}")
    print(f"\n  Of {fn + tp} actual churners, model caught {tp} ({tp/(fn+tp)*100:.1f}%)")
    print(f"  Of {fp + tp} predicted churners, {tp} were correct ({tp/(fp+tp)*100:.1f}%)")
                        

#================================
#FEATURE IMPORTANCE
#================================

def plot_feature_importance(pipeline, feature_names,
                            save_path="data/feature_importance.png",
                            top_n=15):
    """
    Extract and plot feature importance from the trained set.
    Args:
        pipeline: fitted pipeline containing preprocessor and the model
        feature_names: original column names before processing
        save_path: where to save the plot
        top_n: how many top feature to show
    """
    #Extract the fitted processor and model from pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    # Get feature names after transformation
    try:
        transformed_names = preprocessor.get_feature_names_out()
    except AttributeError:
        print("Feature names not available for this preprocessor version.")
        return

    # Get feature importances from the model
    # XGBoost exposes this as feature_importances_
    if not hasattr(model, 'feature_importances_'):
        print("This model does not support feature importance.")
        return

    importances = model.feature_importances_

    # Build a sorted DataFrame for clean plotting
    importance_df = pd.DataFrame({
        'feature': transformed_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x='importance',
        y='feature',
        palette='viridis'
    )
    plt.title(f'Top {top_n} Feature Importances — {pipeline.named_steps["model"].__class__.__name__}',
            fontsize=12)
    plt.xlabel('Importance Score', fontsize=10)
    plt.ylabel('Feature', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")

    # Print top 5 for immediate insight
    print(f"\nTop 5 most important features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")