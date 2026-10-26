"""
Predictive Maintenance & Machine Failure Detection
===================================================
Script: evaluate_accuracy.py
Description: Evaluates and prints the exact accuracy, precision, recall,
             F1-score, ROC-AUC, and PR-AUC of the trained production models
             on the held-out stratified test dataset.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def main():
    print("=" * 75)
    print(" MODEL ACCURACY & PERFORMANCE VERIFICATION")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "ai4i2020.csv")
    model_path = os.path.join(base_dir, "models", "binary_failure_model.joblib")
    diag_model_path = os.path.join(base_dir, "models", "failure_type_model.joblib")

    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model not found at {model_path}. Run 'python train_model.py' first.")
        return

    # 1. Load dataset
    df = pd.read_csv(data_path)
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power_kW"] = (df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi) / 60000.0
    df["Tool_wear_Torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    feature_cols = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "Temp_diff", "Power_kW", "Tool_wear_Torque"
    ]
    X = df[feature_cols]
    y = df["Machine failure"]

    # Stratified test split (identical to training pipeline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # 2. Evaluate Binary Failure Prediction Model
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        pipeline = bundle["pipeline"]
        threshold = float(bundle.get("selected_threshold", 0.50))
        model_name = bundle.get("model_name", "Hist Gradient Boosting (Balanced)")
    else:
        pipeline = bundle
        threshold = 0.50
        model_name = "Trained Classifier"

    probas = pipeline.predict_proba(X_test)[:, 1]
    preds_def = (probas >= 0.50).astype(int)
    preds_opt = (probas >= threshold).astype(int)

    acc_def = accuracy_score(y_test, preds_def)
    acc_opt = accuracy_score(y_test, preds_opt)
    prec_opt = precision_score(y_test, preds_opt, pos_label=1, zero_division=0)
    rec_opt = recall_score(y_test, preds_opt, pos_label=1, zero_division=0)
    f1_opt = f1_score(y_test, preds_opt, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, probas)
    pr_auc = average_precision_score(y_test, probas)
    cm = confusion_matrix(y_test, preds_opt)

    print(f"1. PRIMARY BINARY FAILURE MODEL: {model_name}")
    print(f"   - Test Dataset Size              : {len(y_test):,} samples (68 failures, 1,932 normal)")
    print(f"   - Overall Accuracy (Default 0.50): {acc_def * 100:.2f}% ({acc_def:.4f})")
    print(f"   - Overall Accuracy (Optimal {threshold:.2f}): {acc_opt * 100:.2f}% ({acc_opt:.4f})")
    print(f"   - Failure-Class Precision        : {prec_opt * 100:.2f}% ({prec_opt:.4f})")
    print(f"   - Failure-Class Recall           : {rec_opt * 100:.2f}% ({rec_opt:.4f})")
    print(f"   - Failure-Class F1-Score         : {f1_opt:.4f}")
    print(f"   - ROC-AUC Score                  : {roc_auc:.4f}")
    print(f"   - PR-AUC (Average Precision)     : {pr_auc:.4f}")
    print(f"\n   Confusion Matrix (at threshold {threshold:.2f}):")
    print(f"     * True Negatives (Normal correctly identified)  : {cm[0, 0]:>4}")
    print(f"     * False Positives (False alarms)                : {cm[0, 1]:>4}")
    print(f"     * False Negatives (Missed failures)             : {cm[1, 0]:>4}")
    print(f"     * True Positives (Failures detected in advance) : {cm[1, 1]:>4}")

    print("\n   Full Binary Classification Report:")
    print(classification_report(y_test, preds_opt, target_names=["Normal (0)", "Failure (1)"], digits=4))

    # 3. Evaluate Multiclass Failure-Type Model
    if os.path.exists(diag_model_path):
        diag_pipe = joblib.load(diag_model_path)
        fail_df = df[df["Machine failure"] == 1].copy()

        def assign_mode(r):
            if r["HDF"] == 1: return "Heat Dissipation Failure"
            if r["PWF"] == 1: return "Power Failure"
            if r["OSF"] == 1: return "Overstrain Failure"
            if r["TWF"] == 1: return "Tool Wear Failure"
            return "Unclassified"

        fail_df["Mode"] = fail_df.apply(assign_mode, axis=1)
        known_fails = fail_df[fail_df["Mode"] != "Unclassified"]

        X_f = known_fails[feature_cols]
        y_f = known_fails["Mode"]

        _, X_te_f, _, y_te_f = train_test_split(
            X_f, y_f, test_size=0.25, random_state=RANDOM_STATE, stratify=y_f
        )

        preds_type = diag_pipe.predict(X_te_f)
        acc_type = accuracy_score(y_te_f, preds_type)

        print("-" * 75)
        print("2. MULTICLASS FAILURE-TYPE DIAGNOSTIC MODEL")
        print(f"   - Overall Diagnostic Accuracy    : {acc_type * 100:.2f}% ({acc_type:.4f})")
        print("\n   Per-Failure-Mode Breakdown:")
        print(classification_report(y_te_f, preds_type, digits=4))

    print("=" * 75)

if __name__ == "__main__":
    main()
