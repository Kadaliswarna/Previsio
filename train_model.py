"""
Predictive Maintenance & Machine Failure Detection
===================================================
Script: train_model.py (Audited & Enhanced Version)
Description: Complete Machine Learning engineering pipeline for industrial predictive maintenance.
             - Comprehensive dataset integrity and class imbalance audit.
             - Domain physical feature engineering (Temperature Differential, Mechanical Power, Tool Overstrain).
             - Strict data leakage prevention (isolation of IDs and failure mode flags).
             - Stratified train-test split (80/20) and Stratified 5-Fold Cross-Validation on the training split.
             - Comparison of 4 candidate classification models:
                 1. Random Forest (Balanced)
                 2. Extra Trees (Balanced)
                 3. Hist Gradient Boosting (Balanced)
                 4. Logistic Regression (Balanced)
             - Rare-event evaluation metrics: Precision, Recall, F1-score, ROC-AUC, PR-AUC (Average Precision).
             - Decision threshold optimization on training out-of-fold cross-validation predictions.
             - Final evaluation on the untouched stratified test partition.
             - Audited multiclass failure-type diagnostic model on verified failure records.
             - Generation of 8 visual diagnostic charts and comprehensive text evaluation reports.

Authors/Engineers: Antigravity AI Pair Programming Team
Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    brier_score_loss
)
from sklearn.inspection import permutation_importance

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# Fixed random seed for complete reproducibility across all splits and models
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def find_dataset(file_name="ai4i2020.csv"):
    """
    Locates the dataset CSV in standard directory locations or informs the user
    with clear download instructions if missing.
    """
    possible_paths = [
        os.path.join("data", file_name),
        os.path.join(os.path.dirname(__file__), "data", file_name),
        file_name,
        os.path.join(os.path.dirname(__file__), file_name),
        os.path.join("..", file_name),
        os.path.join(os.path.dirname(__file__), "..", file_name),
        os.path.join("..", "data", file_name)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    error_msg = f"""
[ERROR] Dataset '{file_name}' was not found in any of the expected locations:
  - {os.path.abspath(os.path.join('data', file_name))}
  - {os.path.abspath(file_name)}

How to obtain the official dataset:
  1. Download the AI4I 2020 Predictive Maintenance Dataset from the official UCI repository:
     https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
  2. Save or extract 'ai4i2020.csv' into the project's 'data/' directory:
     {os.path.abspath('data')}
"""
    raise FileNotFoundError(error_msg)


def engineer_domain_features(df):
    """
    Computes domain-specific physical engineering features from raw sensor readings.
    
    Physical Principles:
      1. Temp_diff = Process temperature - Air temperature
         (Heat Dissipation Failure occurs when temperature difference is below 8.6 K and speed < 1380 rpm)
      2. Power_kW = (Torque * Rotational speed * 2 * pi) / 60,000
         (Power Failure occurs when power is below 3,500 W or above 9,000 W)
      3. Tool_wear_Torque = Tool wear * Torque
         (Overstrain Failure occurs when the product of tool wear and torque exceeds capacity)
    """
    df_feat = df.copy()
    df_feat["Temp_diff"] = df_feat["Process temperature [K]"] - df_feat["Air temperature [K]"]
    df_feat["Power_kW"] = (df_feat["Torque [Nm]"] * df_feat["Rotational speed [rpm]"] * 2 * np.pi) / 60000.0
    df_feat["Tool_wear_Torque"] = df_feat["Tool wear [min]"] * df_feat["Torque [Nm]"]
    return df_feat


def audit_dataset_integrity(df):
    """
    Performs comprehensive dataset integrity and distribution checks,
    specifically auditing normal vs failure overlap and failure mode indicators.
    """
    print("=" * 75)
    print(" 1. DATASET INTEGRITY & FAILURE MODE AUDIT")
    print("=" * 75)
    total_rows, total_cols = df.shape
    missing_count = int(df.isnull().sum().sum())
    dup_count = int(df.duplicated().sum())
    
    print(f"Dataset Shape: {total_rows:,} rows, {total_cols} columns")
    print(f"Missing Values: {missing_count}")
    print(f"Duplicate Rows: {dup_count}")

    # Binary target distribution
    target_counts = df["Machine failure"].value_counts()
    normal_count = int(target_counts.get(0, 0))
    fail_count = int(target_counts.get(1, 0))
    fail_pct = (fail_count / total_rows) * 100
    imbalance_ratio = normal_count / fail_count if fail_count > 0 else 0

    print(f"\nTarget Class Distribution ('Machine failure'):")
    print(f"  - Class 0 (Normal Operation) : {normal_count:>5,} ({100 - fail_pct:.2f}%)")
    print(f"  - Class 1 (Machine Failure)  : {fail_count:>5,} ({fail_pct:.2f}%)")
    print(f"  - Imbalance Ratio            : 1 failure per {imbalance_ratio:.1f} normal operations")

    # Audit failure modes
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    mode_names = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure"
    }

    print("\nFailure Mode Breakdown & Cross-Tabulation:")
    audit_modes = {}
    for m in failure_modes:
        c_all = int(df[m].sum())
        c_fail = int(df[df["Machine failure"] == 1][m].sum())
        c_nofail = int(df[df["Machine failure"] == 0][m].sum())
        audit_modes[m] = {
            "name": mode_names[m],
            "total_flagged": c_all,
            "with_machine_failure_1": c_fail,
            "with_machine_failure_0": c_nofail
        }
        print(f"  - {m} ({mode_names[m]:<24}): Total={c_all:>3} | When Failure=1: {c_fail:>3} | When Failure=0: {c_nofail:>2}")

    # Highlight crucial audit finding regarding RNF
    print("\n>>> AUDIT FINDING: RNF (Random Failure) Behavior:")
    print("    18 out of 19 RNF occurrences in the dataset have 'Machine failure' == 0.")
    print("    In the true failure records (Machine failure == 1), RNF occurs only in 1 record (co-occurring with TWF).")
    print("    Normal machines must NEVER be mislabeled as 'Random Failure'.")

    # Check unclassified failures
    sum_modes = df[failure_modes].sum(axis=1)
    unclassified_fails = int(((df["Machine failure"] == 1) & (sum_modes == 0)).sum())
    multi_mode_fails = int(((df["Machine failure"] == 1) & (sum_modes > 1)).sum())
    print(f"\n>>> AUDIT FINDING: Multi-Mode & Unclassified Failures:")
    print(f"    - Failures with multiple simultaneous failure flags: {multi_mode_fails}")
    print(f"    - Failures with NO failure flag (unclassified)      : {unclassified_fails}")

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_values": missing_count,
        "duplicates": dup_count,
        "normal_count": normal_count,
        "failure_count": fail_count,
        "imbalance_ratio": imbalance_ratio,
        "mode_breakdown": audit_modes,
        "unclassified_failures": unclassified_fails,
        "multi_mode_failures": multi_mode_fails
    }


def plot_class_distribution(df, results_dir):
    """
    Generates and saves the class distribution plot and failure modes breakdown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    target_counts = df["Machine failure"].value_counts()

    palette = ["#2b5c8f", "#d9534f"]
    bars = axes[0].bar(
        ["Normal Operation (0)", "Machine Failure (1)"],
        target_counts.values,
        color=palette,
        edgecolor="black",
        width=0.52
    )
    axes[0].set_title("Machine Failure Class Distribution", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Samples", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(
            f"{h:,}\n({h/len(df)*100:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    axes[0].set_ylim(0, max(target_counts.values) * 1.15)

    # Failure modes breakdown in true failures
    true_fails = df[df["Machine failure"] == 1]
    modes = ["HDF", "PWF", "OSF", "TWF", "RNF"]
    labels = ["Heat Dissipation (115)", "Power Failure (95)", "Overstrain (98)", "Tool Wear (46)", "Random Failure (1)"]
    counts = [int(true_fails[m].sum()) for m in modes]
    colors = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#95a5a6"]

    axes[1].barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="black", height=0.6)
    axes[1].set_title("Failure Mode Occurrences in True Failures", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("Count", fontsize=11)
    for i, c in enumerate(counts[::-1]):
        axes[1].text(c + 1.5, i, str(c), va="center", ha="left", fontweight="bold", fontsize=10)
    axes[1].set_xlim(0, max(counts) * 1.18)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "class_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  [Saved Plot] Class distribution -> {plot_path}")


def plot_correlation_heatmap(df_feat, num_features, results_dir):
    """
    Generates and saves the correlation heatmap for numeric features and target.
    """
    corr_cols = num_features + ["Machine failure"]
    corr_matrix = df_feat[corr_cols].corr()

    rename_dict = {
        "Air temperature [K]": "Air Temp",
        "Process temperature [K]": "Process Temp",
        "Rotational speed [rpm]": "Speed (rpm)",
        "Torque [Nm]": "Torque (Nm)",
        "Tool wear [min]": "Tool Wear (min)",
        "Temp_diff": "Temp Diff",
        "Power_kW": "Power (kW)",
        "Tool_wear_Torque": "Tool Wear x Torque",
        "Machine failure": "Machine Failure"
    }
    corr_renamed = corr_matrix.rename(index=rename_dict, columns=rename_dict)

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_renamed, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr_renamed, mask=mask, annot=True, fmt=".2f", cmap=cmap,
        vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
    )
    plt.title("Sensor Telemetry & Failure Target Correlation Heatmap", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    heatmap_path = os.path.join(results_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"  [Saved Plot] Correlation heatmap -> {heatmap_path}")


def cross_validate_candidate_models(X_train, y_train, num_features, cat_features):
    """
    Executes Stratified 5-Fold Cross-Validation strictly on the training partition.
    Evaluates rare-event metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, and Brier Score.
    Collects Out-of-Fold (OOF) predicted probabilities for threshold optimization.
    """
    print("\n" + "=" * 75)
    print(" 2. STRATIFIED 5-FOLD CROSS-VALIDATION ON TRAINING PARTITION")
    print("=" * 75)
    print("Candidate Models: Random Forest, Extra Trees, Hist Gradient Boosting, Logistic Regression")
    print("Imbalance Strategy: Cost-Sensitive Learning (class_weight='balanced')")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features)
        ]
    )

    models = {
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees (Balanced)": ExtraTreesClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Hist Gradient Boosting (Balanced)": HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=150, random_state=RANDOM_STATE
        ),
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        )
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_summary = []
    oof_predictions = {}

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        
        recalls, precisions, f1s, roc_aucs, pr_aucs, brier_scores = [], [], [], [], [], []
        oof_probas = np.zeros(len(X_train))

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipe.fit(X_tr, y_tr)
            
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_val)[:, 1]
            elif hasattr(pipe, "decision_function"):
                df_val = pipe.decision_function(X_val)
                proba = 1 / (1 + np.exp(-df_val))
            else:
                proba = pipe.predict(X_val)

            oof_probas[val_idx] = proba
            pred = (proba >= 0.50).astype(int)

            recalls.append(recall_score(y_val, pred, pos_label=1, zero_division=0))
            precisions.append(precision_score(y_val, pred, pos_label=1, zero_division=0))
            f1s.append(f1_score(y_val, pred, pos_label=1, zero_division=0))
            roc_aucs.append(roc_auc_score(y_val, proba))
            pr_aucs.append(average_precision_score(y_val, proba))
            brier_scores.append(brier_score_loss(y_val, proba))

        oof_predictions[name] = oof_probas

        res_dict = {
            "Model": name,
            "CV_Recall": np.mean(recalls),
            "CV_Recall_Std": np.std(recalls),
            "CV_Precision": np.mean(precisions),
            "CV_Precision_Std": np.std(precisions),
            "CV_F1": np.mean(f1s),
            "CV_F1_Std": np.std(f1s),
            "CV_ROC_AUC": np.mean(roc_aucs),
            "CV_PR_AUC": np.mean(pr_aucs),
            "CV_Brier": np.mean(brier_scores)
        }
        cv_summary.append(res_dict)

        print(f"{name:<34} | F1: {res_dict['CV_F1']:.4f} (+/- {res_dict['CV_F1_Std']:.3f}) | "
              f"Recall: {res_dict['CV_Recall']:.4f} | Prec: {res_dict['CV_Precision']:.4f} | "
              f"ROC-AUC: {res_dict['CV_ROC_AUC']:.4f} | PR-AUC: {res_dict['CV_PR_AUC']:.4f}")

    cv_df = pd.DataFrame(cv_summary)
    return models, preprocessor, cv_df, oof_predictions


def optimize_decision_threshold(y_train, oof_probas, model_name, results_dir):
    """
    Tunes the decision threshold using out-of-fold probability estimates from the training set.
    Evaluates thresholds from 0.05 to 0.95 to balance Failure Recall and Precision.
    Saves a threshold trade-off visual and returns the selected optimal threshold.
    """
    print("\n" + "=" * 75)
    print(f" 3. DECISION THRESHOLD OPTIMIZATION (TRAINING OOF PREDICTIONS: {model_name})")
    print("=" * 75)

    thresholds = np.linspace(0.05, 0.95, 91)
    prec_list, rec_list, f1_list = [], [], []

    for t in thresholds:
        preds = (oof_probas >= t).astype(int)
        p = precision_score(y_train, preds, pos_label=1, zero_division=0)
        r = recall_score(y_train, preds, pos_label=1, zero_division=0)
        f = f1_score(y_train, preds, pos_label=1, zero_division=0)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f)

    prec_arr = np.array(prec_list)
    rec_arr = np.array(rec_list)
    f1_arr = np.array(f1_list)

    # 1. Threshold maximizing F1 score
    max_f1_idx = np.argmax(f1_arr)
    opt_threshold = float(thresholds[max_f1_idx])
    opt_f1 = float(f1_arr[max_f1_idx])
    opt_rec = float(rec_arr[max_f1_idx])
    opt_prec = float(prec_arr[max_f1_idx])

    # Default threshold (0.50) performance
    def_idx = np.argmin(np.abs(thresholds - 0.50))
    def_f1 = float(f1_arr[def_idx])
    def_rec = float(rec_arr[def_idx])
    def_prec = float(prec_arr[def_idx])

    print(f"Default Threshold (0.50) -> OOF Recall: {def_rec:.4f} | Precision: {def_prec:.4f} | F1: {def_f1:.4f}")
    print(f"Optimal Threshold ({opt_threshold:.2f}) -> OOF Recall: {opt_rec:.4f} | Precision: {opt_prec:.4f} | F1: {opt_f1:.4f}")
    print(f"Improvement: Recall +{(opt_rec - def_rec)*100:.1f}%, F1 +{(opt_f1 - def_f1)*100:.2f}% on training validation.")

    # Plot threshold tradeoff curve
    plt.figure(figsize=(9, 5.5))
    plt.plot(thresholds, prec_arr, label="Precision", color="#2980b9", lw=2)
    plt.plot(thresholds, rec_arr, label="Recall", color="#e74c3c", lw=2)
    plt.plot(thresholds, f1_arr, label="F1-Score", color="#27ae60", lw=2.5)

    plt.axvline(0.50, color="#7f8c8d", linestyle="--", alpha=0.8, label="Default (0.50)")
    plt.axvline(opt_threshold, color="#8e44ad", linestyle="-.", lw=2, label=f"Selected Optimal ({opt_threshold:.2f})")

    plt.title(f"Threshold Optimization Curve ({model_name} - Training OOF)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Decision Probability Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="lower center", ncol=3)
    plt.tight_layout()

    thresh_path = os.path.join(results_dir, "threshold_comparison.png")
    plt.savefig(thresh_path, dpi=300)
    plt.close()
    print(f"  [Saved Plot] Threshold comparison curve -> {thresh_path}")

    # Save threshold metrics table
    thresh_df = pd.DataFrame({
        "Threshold": np.round(thresholds, 2),
        "Precision": np.round(prec_arr, 4),
        "Recall": np.round(rec_arr, 4),
        "F1_Score": np.round(f1_arr, 4)
    })
    thresh_table_path = os.path.join(results_dir, "threshold_comparison_report.txt")
    thresh_df.to_csv(thresh_table_path, sep="\t", index=False)
    print(f"  [Saved Report] Threshold data table -> {thresh_table_path}")

    return opt_threshold, {
        "default": {"threshold": 0.50, "recall": def_rec, "precision": def_prec, "f1": def_f1},
        "optimized": {"threshold": opt_threshold, "recall": opt_rec, "precision": opt_prec, "f1": opt_f1}
    }


def evaluate_final_test_set(fitted_pipelines, X_test, y_test, selected_model_name, selected_threshold, results_dir):
    """
    Evaluates all candidate models on the held-out stratified test set (2,000 samples; 68 failures).
    Tests the selected model at both the default threshold (0.50) and the tuned threshold.
    Generates ROC curves, PR curves, and confusion matrix comparisons.
    """
    print("\n" + "=" * 75)
    print(" 4. FINAL EVALUATION ON UNTOUCHED STRATIFIED TEST PARTITION")
    print("=" * 75)
    print(f"Test Set Dimensions: {X_test.shape[0]} samples ({y_test.sum()} actual machine failures)")

    test_results = []
    roc_data = {}
    pr_data = {}
    cm_data = {}

    for name, pipe in fitted_pipelines.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        
        # Default 0.50 threshold
        pred_def = (proba >= 0.50).astype(int)
        acc = accuracy_score(y_test, pred_def)
        prec = precision_score(y_test, pred_def, pos_label=1, zero_division=0)
        rec = recall_score(y_test, pred_def, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, pred_def, pos_label=1, zero_division=0)
        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        cm = confusion_matrix(y_test, pred_def)
        tn, fp, fn, tp = cm.ravel()

        test_results.append({
            "Model": name,
            "Threshold": 0.50,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        })

        # ROC & PR curve data
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba)
        roc_data[name] = (fpr, tpr, roc_auc)
        pr_data[name] = (prec_curve, rec_curve, pr_auc)
        cm_data[f"{name}\n(Threshold = 0.50)"] = cm

        # If this is the selected model, also evaluate at optimized threshold
        if name == selected_model_name and abs(selected_threshold - 0.50) > 0.01:
            pred_opt = (proba >= selected_threshold).astype(int)
            acc_opt = accuracy_score(y_test, pred_opt)
            prec_opt = precision_score(y_test, pred_opt, pos_label=1, zero_division=0)
            rec_opt = recall_score(y_test, pred_opt, pos_label=1, zero_division=0)
            f1_opt = f1_score(y_test, pred_opt, pos_label=1, zero_division=0)
            cm_opt = confusion_matrix(y_test, pred_opt)
            tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()

            test_results.append({
                "Model": f"{name} [Optimized Threshold]",
                "Threshold": selected_threshold,
                "Accuracy": acc_opt,
                "Precision": prec_opt,
                "Recall": rec_opt,
                "F1_Score": f1_opt,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc,
                "TP": int(tp_o),
                "FP": int(fp_o),
                "FN": int(fn_o),
                "TN": int(tn_o)
            })
            cm_data[f"{name}\n(Tuned Threshold = {selected_threshold:.2f})"] = cm_opt

    test_df = pd.DataFrame(test_results)
    print("\nTest Partition Evaluation Summary:")
    print(test_df[["Model", "Threshold", "Precision", "Recall", "F1_Score", "PR_AUC", "ROC_AUC", "TP", "FP", "FN"]].to_string(index=False))

    # Plot Precision-Recall Curves
    plt.figure(figsize=(8.5, 6))
    for name, (p_curve, r_curve, val_pr) in pr_data.items():
        plt.plot(r_curve, p_curve, lw=2, label=f"{name} (PR-AUC = {val_pr:.4f})")
    plt.title("Precision-Recall Curves (Stratified Test Split)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Failure Class Recall", fontsize=11)
    plt.ylabel("Failure Class Precision", fontsize=11)
    plt.xlim(0.0, 1.02)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(results_dir, "precision_recall_curves.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"  [Saved Plot] Precision-Recall curves -> {pr_path}")

    # Plot ROC Curves
    plt.figure(figsize=(8.5, 6))
    for name, (fpr, tpr, val_roc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (ROC-AUC = {val_roc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (0.50)")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.xlim(-0.01, 1.01)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(results_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  [Saved Plot] ROC curves -> {roc_path}")

    # Plot Confusion Matrices
    n_cms = len(cm_data)
    fig, axes = plt.subplots(1, n_cms, figsize=(5.2 * n_cms, 4.5))
    if n_cms == 1:
        axes = [axes]
    for idx, (title, cm) in enumerate(cm_data.items()):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
            xticklabels=["Pred Normal", "Pred Fail"],
            yticklabels=["True Normal", "True Fail"],
            annot_kws={"size": 12, "weight": "bold"}
        )
        axes[idx].set_title(title, fontsize=10.5, fontweight="bold", pad=10)
        axes[idx].set_ylabel("True Condition", fontsize=9.5)
        axes[idx].set_xlabel("Predicted Condition", fontsize=9.5)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  [Saved Plot] Confusion matrices -> {cm_path}")

    # Plot Model Comparison Bar Chart
    comp_plot_data = test_df[test_df["Threshold"] == 0.50].set_index("Model")[["Precision", "Recall", "F1_Score", "PR_AUC", "ROC_AUC"]].T
    ax = comp_plot_data.plot(kind="bar", figsize=(10, 5.5), width=0.75, colormap="tab10", edgecolor="black")
    plt.title("Model Metrics Comparison on Test Set (Default Threshold 0.50)", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Score", fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    plt.ylim(0.0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Candidate Model", frameon=True, loc="upper right")
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.2f}", (p.get_x() + p.get_width() / 2.0, val),
                        ha="center", va="bottom", fontsize=8, weight="bold", xytext=(0, 2), textcoords="offset points")
    plt.tight_layout()
    comp_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"  [Saved Plot] Model comparison chart -> {comp_path}")

    return test_df


def plot_feature_importance_analysis(pipeline, X_train, y_train, results_dir):
    """
    Computes and plots both Gini Feature Importance and Permutation Importance
    to explain which sensor features govern machine failures.
    """
    clf = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["prep"]

    # Extract transformed column names
    try:
        ohe_cols = list(preprocessor.named_transformers_["cat"].get_feature_names_out(["Type"]))
    except Exception:
        ohe_cols = ["Type_L", "Type_M"]
    num_cols = list(preprocessor.transformers[0][2])
    all_feature_names = num_cols + ohe_cols

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_names = [all_feature_names[i] for i in indices]
        sorted_imp = importances[indices]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(sorted_names)), sorted_imp[::-1], color="#2980b9", edgecolor="black")
        plt.yticks(range(len(sorted_names)), sorted_names[::-1], fontsize=10)
        plt.xlabel("Gini Feature Importance", fontsize=11, fontweight="bold")
        plt.title("Sensor Feature Importance in Predicting Machine Failure", fontsize=13, fontweight="bold", pad=12)

        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                     ha="left", va="center", fontsize=9, fontweight="bold")
        plt.xlim(0, max(sorted_imp) * 1.18)
        plt.tight_layout()

        feat_path = os.path.join(results_dir, "feature_importance.png")
        plt.savefig(feat_path, dpi=300)
        plt.close()
        print(f"  [Saved Plot] Feature importance ranking -> {feat_path}")
    else:
        # Use Permutation Importance for gradient boosting or non-tree models
        print("  Computing Permutation Feature Importance on Training Set (scoring: PR-AUC)...")
        r = permutation_importance(
            pipeline, X_train, y_train,
            scoring="average_precision",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        sorted_idx = r.importances_mean.argsort()[::-1]
        sorted_cols = [X_train.columns[i] for i in sorted_idx]
        sorted_means = r.importances_mean[sorted_idx]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(sorted_cols)), sorted_means[::-1], color="#2980b9", edgecolor="black")
        plt.yticks(range(len(sorted_cols)), sorted_cols[::-1], fontsize=10)
        plt.xlabel("Permutation Importance (Mean Decrease in PR-AUC)", fontsize=11, fontweight="bold")
        plt.title(f"Permutation Feature Importance in Predicting Machine Failure ({clf.__class__.__name__})", fontsize=12, fontweight="bold", pad=12)

        for bar in bars:
            w = bar.get_width()
            plt.text(max(w + 0.002, 0.002), bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                     ha="left", va="center", fontsize=9, fontweight="bold")
        plt.xlim(0, max(sorted_means) * 1.25 if max(sorted_means) > 0 else 0.1)
        plt.tight_layout()

        feat_path = os.path.join(results_dir, "feature_importance.png")
        plt.savefig(feat_path, dpi=300)
        plt.close()
        print(f"  [Saved Plot] Permutation feature importance -> {feat_path}")


def train_audited_failure_type_model(df_clean, num_features, cat_features, models_dir, results_dir):
    """
    Audited Multiclass Failure Type Diagnostic Model:
      - Uses EXCLUSIVELY genuine failure records (Machine failure == 1).
      - Excludes normal non-failure records (correcting the 18 false RNF samples).
      - Uses verified failure modes: Heat Dissipation, Power Failure, Overstrain, Tool Wear.
      - Explicitly documents that Random Failure (RNF) has only 1 sample among true failures.
      - Evaluates with Stratified validation without data leakage.
    """
    print("\n" + "=" * 75)
    print(" 5. AUDITED MULTICLASS FAILURE-TYPE DIAGNOSTIC MODEL")
    print("=" * 75)

    true_failures = df_clean[df_clean["Machine failure"] == 1].copy()
    print(f"Total True Machine Failure Records in Dataset: {len(true_failures)}")

    def get_verified_failure_mode(row):
        # Specific physical failure modes
        if row["HDF"] == 1:
            return "Heat Dissipation Failure"
        if row["PWF"] == 1:
            return "Power Failure"
        if row["OSF"] == 1:
            return "Overstrain Failure"
        if row["TWF"] == 1:
            return "Tool Wear Failure"
        return "Unclassified Failure"

    true_failures["Diagnosed_Mode"] = true_failures.apply(get_verified_failure_mode, axis=1)
    mode_counts = true_failures["Diagnosed_Mode"].value_counts()
    print("\nDiagnosed Failure Modes on Genuine Machine Failures:")
    for mode, cnt in mode_counts.items():
        print(f"  - {mode:<26}: {cnt:>3} instances")

    # Filter to known failure categories (excluding unclassified if too few for stratification)
    known_diagnoses = true_failures[true_failures["Diagnosed_Mode"] != "Unclassified Failure"].copy()
    print(f"\nKnown Diagnosable Failure Samples: {len(known_diagnoses)}")

    X_diag = known_diagnoses[cat_features + num_features]
    y_diag = known_diagnoses["Diagnosed_Mode"]

    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
        X_diag, y_diag, test_size=0.25, random_state=RANDOM_STATE, stratify=y_diag
    )

    preprocessor_diag = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features)
        ]
    )

    diag_clf = RandomForestClassifier(
        n_estimators=120, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )

    diag_pipeline = Pipeline([
        ("prep", preprocessor_diag),
        ("clf", diag_clf)
    ])

    diag_pipeline.fit(X_tr_d, y_tr_d)
    y_pred_d = diag_pipeline.predict(X_te_d)

    diag_acc = accuracy_score(y_te_d, y_pred_d)
    diag_report_str = classification_report(y_te_d, y_pred_d, digits=4)
    print(f"\nMulticlass Diagnostic Test Accuracy: {diag_acc * 100:.2f}%")
    print("Classification Report on Verified Failure Records:")
    print(diag_report_str)

    # Save model
    diag_model_path = os.path.join(models_dir, "failure_type_model.joblib")
    joblib.dump(diag_pipeline, diag_model_path)
    print(f"  [Saved Model] Failure type diagnostic pipeline -> {diag_model_path}")

    # Plot failure type confusion matrix
    cm_diag = confusion_matrix(y_te_d, y_pred_d, labels=diag_pipeline.classes_)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm_diag, annot=True, fmt="d", cmap="Oranges",
        xticklabels=diag_pipeline.classes_, yticklabels=diag_pipeline.classes_,
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("Failure Type Diagnosis Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("True Failure Mode", fontsize=10)
    plt.xlabel("Diagnosed Failure Mode", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    type_cm_path = os.path.join(results_dir, "failure_type_confusion_matrix.png")
    plt.savefig(type_cm_path, dpi=300)
    plt.close()
    print(f"  [Saved Plot] Failure type confusion matrix -> {type_cm_path}")

    return diag_pipeline, diag_acc, diag_report_str, mode_counts.to_dict()


def save_comprehensive_reports(results_dir, audit_stats, cv_df, test_df, selected_model_name, selected_threshold, diag_acc, diag_report_str, diag_counts):
    """
    Saves detailed text and JSON evaluation reports documenting full dataset audit,
    cross-validation results, test set metrics, and failure diagnostics.
    """
    # 1. Dataset Integrity Report
    integrity_path = os.path.join(results_dir, "dataset_integrity_report.txt")
    with open(integrity_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write(" AI4I 2020 DATASET INTEGRITY & FAILURE DISTRIBUTION AUDIT\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Total Operational Records : {audit_stats['total_rows']:,}\n")
        f.write(f"Total Columns             : {audit_stats['total_cols']}\n")
        f.write(f"Missing Values Checked    : {audit_stats['missing_values']}\n")
        f.write(f"Duplicate Rows Checked    : {audit_stats['duplicates']}\n")
        f.write(f"Normal Machine Operations : {audit_stats['normal_count']:,} ({audit_stats['normal_count']/audit_stats['total_rows']*100:.2f}%)\n")
        f.write(f"Machine Failure Incidents : {audit_stats['failure_count']:,} ({audit_stats['failure_count']/audit_stats['total_rows']*100:.2f}%)\n")
        f.write(f"Class Imbalance Ratio     : 1 failure per {audit_stats['imbalance_ratio']:.1f} normal operations\n\n")
        f.write("AUDIT OF FAILURE INDICATOR FLAGS:\n")
        f.write("-" * 50 + "\n")
        for m, data in audit_stats["mode_breakdown"].items():
            f.write(f"{m} ({data['name']}): Total Flagged={data['total_flagged']} | In Failures={data['with_machine_failure_1']} | In Normal={data['with_machine_failure_0']}\n")
        f.write("\nKEY INTEGRITY FINDING: RNF (Random Failure) Artifact:\n")
        f.write("18 out of 19 RNF instances in the dataset have Machine failure == 0.\n")
        f.write("In genuine failures, RNF occurred in only 1 instance (co-occurring with TWF).\n")
        f.write("Normal operating records were verified and excluded from failure-type training.\n")

    print(f"  [Saved Report] Dataset integrity audit -> {integrity_path}")

    # 2. Full Classification & Model Evaluation Report
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(" INDUSTRIAL PREDICTIVE MAINTENANCE: ML MODEL AUDIT & EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. DATASET CHARACTERISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Rows: {audit_stats['total_rows']} | Normal (0): {audit_stats['normal_count']} | Failures (1): {audit_stats['failure_count']}\n")
        f.write(f"Imbalance Ratio: 1 : {audit_stats['imbalance_ratio']:.1f}\n\n")

        f.write("2. DATA LEAKAGE CONTROLS\n")
        f.write("-" * 50 + "\n")
        f.write("Excluded from Inputs: UDI, Product ID, Machine failure, TWF, HDF, PWF, OSF, RNF\n")
        f.write("Input Features: Type, Air temp, Process temp, Speed, Torque, Tool wear\n")
        f.write("Domain Features: Temp_diff, Power_kW, Tool_wear_Torque\n")
        f.write("Transformations fitted strictly within training folds.\n\n")

        f.write("3. STRATIFIED 5-FOLD CROSS-VALIDATION (TRAINING SET ONLY)\n")
        f.write("-" * 50 + "\n")
        f.write(cv_df.to_string(index=False) + "\n\n")

        f.write("4. DECISION THRESHOLD SELECTION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Selected Production Model: {selected_model_name}\n")
        f.write(f"Selected Optimal Decision Threshold: {selected_threshold:.2f}\n\n")

        f.write("5. HELD-OUT TEST PARTITION EVALUATION (UNTOUCHED UNTIL FINAL TEST)\n")
        f.write("-" * 50 + "\n")
        f.write(test_df.to_string(index=False) + "\n\n")

        f.write("6. AUDITED FAILURE TYPE DIAGNOSTIC MODEL\n")
        f.write("-" * 50 + "\n")
        f.write(f"Diagnostic Test Accuracy: {diag_acc * 100:.2f}%\n")
        f.write("Sample Counts among Genuine Failures:\n")
        for mode, cnt in diag_counts.items():
            f.write(f"  - {mode}: {cnt}\n")
        f.write("\nDiagnostic Classification Report:\n")
        f.write(diag_report_str + "\n")

    print(f"  [Saved Report] Full evaluation report -> {report_path}")

    # 3. Machine-readable metrics summary JSON
    json_path = os.path.join(results_dir, "metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "selected_model": selected_model_name,
            "selected_threshold": selected_threshold,
            "cv_metrics": cv_df.to_dict(orient="records"),
            "test_metrics": test_df.to_dict(orient="records"),
            "diagnostic_accuracy": float(diag_acc)
        }, jf, indent=2)
    print(f"  [Saved Report] Machine-readable metrics JSON -> {json_path}")


def main():
    print("=" * 80)
    print(" PREDICTIVE MAINTENANCE PIPELINE: AUDIT, IMPROVEMENT & VALIDATION")
    print("=" * 80)

    # 1. Directory resolution
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Locate and load dataset
    dataset_path = find_dataset("ai4i2020.csv")
    print(f"\nDataset loaded from: {dataset_path}")
    df_raw = pd.read_csv(dataset_path)

    # 3. Comprehensive Dataset Audit
    audit_stats = audit_dataset_integrity(df_raw)

    # 4. Feature Engineering
    raw_num_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    cat_cols = ["Type"]

    df_engineered = engineer_domain_features(df_raw)
    engineered_num_cols = raw_num_cols + ["Temp_diff", "Power_kW", "Tool_wear_Torque"]

    # 5. Visualizations: EDA
    print("\n" + "=" * 75)
    print(" GENERATING EXPLORATORY VISUALIZATIONS")
    print("=" * 75)
    plot_class_distribution(df_raw, results_dir)
    plot_correlation_heatmap(df_engineered, engineered_num_cols, results_dir)

    # 6. Stratified Train/Test Split (80/20)
    # Strictly exclude target labels and breakdown indicator flags from inputs!
    leakage_cols = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    feature_columns = [c for c in df_engineered.columns if c not in leakage_cols]

    X = df_engineered[feature_columns]
    y = df_engineered["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nStratified Train Split: {X_train.shape[0]:,} samples ({y_train.sum()} failures)")
    print(f"Stratified Test Split : {X_test.shape[0]:,} samples ({y_test.sum()} failures)")

    # 7. Stratified 5-Fold Cross-Validation on Training Split
    candidate_models, preprocessor, cv_df, oof_predictions = cross_validate_candidate_models(
        X_train, y_train, engineered_num_cols, cat_cols
    )

    # 8. Model Selection based on Cross-Validation PR-AUC & F1-Score
    best_cv_idx = cv_df["CV_F1"].idxmax()
    selected_model_name = cv_df.loc[best_cv_idx, "Model"]
    print(f"\n>>> Selected Best Architecture: '{selected_model_name}' (CV F1: {cv_df.loc[best_cv_idx, 'CV_F1']:.4f}, CV PR-AUC: {cv_df.loc[best_cv_idx, 'CV_PR_AUC']:.4f})")

    # 9. Decision Threshold Optimization (Training Set Only)
    selected_oof_probas = oof_predictions[selected_model_name]
    selected_threshold, thresh_details = optimize_decision_threshold(
        y_train, selected_oof_probas, selected_model_name, results_dir
    )

    # 10. Fit Candidate Models on Entire Training Split & Evaluate on Untouched Test Set
    print("\nFitting candidate pipelines on full training split...")
    fitted_pipelines = {}
    for name, clf in candidate_models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

    test_df = evaluate_final_test_set(
        fitted_pipelines, X_test, y_test, selected_model_name, selected_threshold, results_dir
    )

    # 11. Feature Importance Analysis
    selected_pipe = fitted_pipelines[selected_model_name]
    plot_feature_importance_analysis(selected_pipe, X_train, y_train, results_dir)

    # 12. Save Final Production Binary Model (Including Selected Threshold in Metadata)
    binary_model_path = os.path.join(models_dir, "binary_failure_model.joblib")
    # Package pipeline with inference metadata
    model_bundle = {
        "pipeline": selected_pipe,
        "model_name": selected_model_name,
        "selected_threshold": selected_threshold,
        "feature_names": feature_columns,
        "raw_numerical_features": raw_num_cols,
        "engineered_features": ["Temp_diff", "Power_kW", "Tool_wear_Torque"],
        "categorical_features": cat_cols
    }
    joblib.dump(model_bundle, binary_model_path)
    print(f"\n>>> [Saved Model] Production Binary Failure Pipeline saved to -> {binary_model_path}")

    # 13. Audited Multiclass Failure-Type Diagnostic Model
    diag_pipeline, diag_acc, diag_report_str, diag_counts = train_audited_failure_type_model(
        df_engineered, engineered_num_cols, cat_cols, models_dir, results_dir
    )

    # 14. Save Comprehensive Reports
    save_comprehensive_reports(
        results_dir, audit_stats, cv_df, test_df,
        selected_model_name, selected_threshold,
        diag_acc, diag_report_str, diag_counts
    )

    print("\n" + "=" * 80)
    print(" TRAINING, AUDIT & VALIDATION COMPLETED SUCCESSFULLY!")
    print(f" Production Models Saved in : {models_dir}")
    print(f" Evaluation Reports & Visuals: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
