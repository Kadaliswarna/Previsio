# Predictive Maintenance & Machine Failure Detection (Audited & Production-Grade)

An audited, rigorously evaluated Machine Learning system for industrial predictive maintenance using sensor telemetry from the **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository).

This project detects incipient machine failures, categorizes equipment health into operational risk tiers, optimizes decision thresholds for rare-event detection, and diagnoses the specific physical failure mechanism (*Heat Dissipation Failure*, *Power Failure*, *Overstrain Failure*, *Tool Wear Failure*).

---

## 1. Executive Summary & Audit Overview

### Initial Reported Results vs. Independently Audited Findings

| Metric / Aspect | Initial Reported Claim | Independent Senior ML Audit Finding |
| :--- | :--- | :--- |
| **Dataset Dimensions** | 10,000 rows, 14 columns | **Verified**: 10,000 rows, 14 columns, 0 missing values, 0 duplicate rows. |
| **Machine Failure Count** | 339 failure records (3.39%) | **Verified**: 339 failures, 9,661 normal operations ($1:28.5$ class imbalance). |
| **Binary Model Validation** | Evaluated only on a single 80/20 split | **Improved**: Added **Stratified 5-Fold Cross-Validation** on the training partition ($n=8,000$) before touching the final test set ($n=2,000$). |
| **Rare-Event Evaluation** | Evaluated on accuracy & default threshold ($0.50$) | **Improved**: Evaluated **PR-AUC (Average Precision)**, ROC-AUC, Brier score, and **Threshold Optimization** on out-of-fold predictions. |
| **Multiclass Failure Diagnosis** | Claimed 98.85% accuracy on 5 classes (including RNF) | **AUDIT FIX**: Found that 18 out of 19 `RNF` (Random Failure) records were **normal non-failure machines (`Machine failure == 0`)**. Only 1 true failure had `RNF`. Training on normal records mislabeled as "Random Failure" was eliminated; the audited model trains strictly on verified failure instances ($n=330$). |
| **Data Leakage Controls** | `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` excluded | **Verified & Hardened**: Encoders, scalers, and domain engineering are fitted exclusively inside training folds. |

---

## 2. Dataset Integrity & Failure Mode Breakdown

### Binary Target (`Machine failure`)
- **Normal Operation (`0`)**: 9,661 units (96.61%)
- **Machine Failure (`1`)**: 339 units (3.39%)
- **Imbalance Ratio**: $1 : 28.5$

### Failure Mode Breakdown in True Machine Failures (`Machine failure == 1`)
- **Heat Dissipation Failure (HDF)**: 115 instances
- **Power Failure (PWF)**: 95 instances
- **Overstrain Failure (OSF)**: 98 instances
- **Tool Wear Failure (TWF)**: 46 instances
- **Random Failure (RNF)**: **1 instance** (co-occurring with TWF)
- **Multi-Mode Failures**: 24 machines experience dual or triple failures simultaneously (e.g., PWF + OSF).
- **Unclassified Failures**: 9 machines had `Machine failure == 1` with zero failure mode flags set.

> [!WARNING]
> **Audit Finding on RNF**: The UCI AI4I 2020 dataset documentation states that RNF represents a synthetic 0.1% random failure noise process. In 18 of the 19 rows where `RNF = 1`, the machine tool did **not** shut off (`Machine failure = 0`). Treating those 18 normal records as a distinct "Random Failure" class mislabels normal machines and creates an artificial test set with zero genuine failure foundation. The audited pipeline restricts failure-type training strictly to genuine failure instances.

---

## 3. Physical Domain Feature Engineering & Leakage Prevention

### Excluded Columns (Strict Leakage Prevention)
The following columns are **never** provided to the binary classification model:
- `UDI`, `Product ID` (Identifiers)
- `Machine failure` (Ground truth target)
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (Target-derived failure indicator flags)

### Engineered Physical Features
Derived strictly from the individual machine's instantaneous sensor readings:
1. **Temperature Difference ($\Delta T$)**:
   $$\Delta T = \text{Process Temperature [K]} - \text{Air Temperature [K]}$$
   *Physical Mechanism*: Heat Dissipation Failure (HDF) occurs when $\Delta T < 8.6\text{ K}$ at rotational speeds $< 1380\text{ rpm}$.
2. **Mechanical Power ($P_{\text{mech}}$)**:
   $$P_{\text{mech}} = \frac{\text{Torque [Nm]} \times \text{Rotational Speed [rpm]} \times 2\pi}{60,000}\quad [\text{kW}]$$
   *Physical Mechanism*: Power Failure (PWF) occurs when mechanical power draw falls below $3.5\text{ kW}$ or exceeds $9.0\text{ kW}$.
3. **Tool Overstrain Index ($I_{\text{OS}}$)**:
   $$I_{\text{OS}} = \text{Tool Wear [min]} \times \text{Torque [Nm]}$$
   *Physical Mechanism*: Overstrain Failure (OSF) occurs when high torque load combines with excessive blade wear.

---

## 4. Model Training & Cross-Validation Results

### Stratified 5-Fold Cross-Validation on Training Split ($n=8,000$; 271 Failures)
All candidate models use cost-sensitive balanced weighting (`class_weight='balanced'`) to prevent majority-class bias.

| Candidate Model | CV F1-Score | CV Recall | CV Precision | CV PR-AUC | CV ROC-AUC | CV Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hist Gradient Boosting (Balanced)** | **0.8614** ($\pm0.021$) | **82.65%** | **89.99%** | **0.8768** | **0.9698** | **0.0076** |
| **Random Forest (Balanced)** | 0.8567 ($\pm0.036$) | 76.74% | 97.15% | 0.8976 | 0.9805 | 0.0071 |
| **Extra Trees (Balanced)** | 0.7338 ($\pm0.015$) | 74.18% | 72.69% | 0.7965 | 0.9682 | 0.0150 |
| **Logistic Regression (Balanced)** | 0.2716 ($\pm0.011$) | 84.15% | 16.21% | 0.4587 | 0.9233 | 0.1197 |

> [!NOTE]
> **Hist Gradient Boosting (Balanced)** was selected as the top architecture because it maximized balanced cross-validation F1-score (**0.8614**) while delivering high failure recall (**82.65%**) and strong precision (**89.99%**).

---

## 5. Decision Threshold Optimization

Using out-of-fold probability predictions across the training set (leaving the final test partition completely untouched):
- **Default Threshold ($0.50$)**: OOF Recall = $82.66\%$, OOF Precision = $89.96\%$, OOF F1 = $0.8615$
- **Selected Optimal Threshold ($0.53$)**: OOF Recall = $82.66\%$, OOF Precision = $90.69\%$, OOF F1 = **$0.8649$**

---

## 6. Final Evaluation on Untouched Stratified Test Partition

Evaluated on the held-out 2,000-sample test set (1,932 normal, 68 failures):

| Model | Decision Threshold | Test Precision | Test Recall | Test F1-Score | Test PR-AUC | Test ROC-AUC | True Positives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hist Gradient Boosting (Balanced)** | **0.53** *(Optimal)* | **91.80%** | **82.35%** | **0.8682** | **0.8976** | **0.9782** | **56** | **5** | **12** |
| **Hist Gradient Boosting (Balanced)** | 0.50 *(Default)* | 91.80% | 82.35% | 0.8682 | 0.8976 | 0.9782 | 56 | 5 | 12 |
| **Random Forest (Balanced)** | 0.50 | 96.49% | 80.88% | 0.8800 | 0.8785 | 0.9869 | 55 | 2 | 13 |
| **Extra Trees (Balanced)** | 0.50 | 68.49% | 73.53% | 0.7092 | 0.7641 | 0.9673 | 50 | 23 | 18 |
| **Logistic Regression (Balanced)** | 0.50 | 17.96% | 88.24% | 0.2985 | 0.4296 | 0.9385 | 60 | 274 | 8 |

---

## 7. Audited Multiclass Failure-Type Diagnostic Model

Trained exclusively on verified machine failures with known diagnoses ($n=330$; Test split: $n=83$):
- **Overall Diagnostic Accuracy**: **98.80%**
- **Macro F1-Score**: **0.9894**

```text
                          precision    recall  f1-score   support

Heat Dissipation Failure     0.9667    1.0000    0.9831        29
      Overstrain Failure     1.0000    0.9500    0.9744        20
           Power Failure     1.0000    1.0000    1.0000        23
       Tool Wear Failure     1.0000    1.0000    1.0000        11

                accuracy                         0.9880        83
               macro avg     0.9917    0.9875    0.9894        83
            weighted avg     0.9884    0.9880    0.9879        83
```

---

## 8. Saved Visualizations and Reports

All files are located in `results/`:
- `class_distribution.png`: Binary class distribution and breakdown of verified failure modes.
- `correlation_heatmap.png`: High-resolution Pearson correlation matrix.
- `model_comparison.png`: Bar chart comparing models across Precision, Recall, F1, PR-AUC, and ROC-AUC.
- `confusion_matrices.png`: Side-by-side confusion matrix heatmaps on the test set.
- `precision_recall_curves.png`: Precision-Recall curves displaying PR-AUC for each candidate.
- `roc_curves.png`: Receiver Operating Characteristic curves displaying ROC-AUC.
- `feature_importance.png`: Permutation feature importance ranking on PR-AUC.
- `threshold_comparison.png`: Trade-off curves across decision thresholds (0.05 to 0.95).
- `failure_type_confusion_matrix.png`: Confusion matrix for the 4 verified failure modes.
- `dataset_integrity_report.txt`: Full text audit of dataset and failure modes.
- `classification_report.txt`: Comprehensive model evaluation report.
- `threshold_comparison_report.txt`: Tabular data of threshold sweep.
- `metrics_summary.json`: Machine-readable evaluation metrics.

---

## 9. Windows VS Code Terminal Commands

Follow these exact commands in your Windows VS Code terminal (PowerShell):

### Step 1: Open Terminal & Navigate to Project
```powershell
cd c:\Users\kadal\Downloads\ai4i+2020+predictive+maintenance+dataset\predictive-maintenance-ml
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Training, Cross-Validation & Auditing
```powershell
python train_model.py
```

```

### Step 5: Launch the Interactive Web Dashboard

```powershell
python -m streamlit run app.py
```
This launches the modern industrial predictive maintenance dashboard on `http://localhost:8501`:
- **Dashboard Overview**: Fleet KPI cards, baseline vs verified metrics comparison, operating distribution pie chart, and failure mode frequencies.
- **Machine Health Prediction**: Single equipment form with sliders/inputs, benchmark presets (Normal, Thermal Defect, Overstrain, Spindle Surge), failure probability gauge, and maintenance guidance.
- **Failure-Type Diagnosis**: Multiclass diagnostic engine evaluating root-cause failure mode probabilities.
- **Telemetry Analytics**: Interactive Plotly scatter plots, sensor correlation heatmaps, and confusion matrix visualizer.

### Step 6: Run CLI & Batch Predictions

```powershell
# Run built-in industrial benchmark scenarios (Healthy, Thermal, Overstrain, Power)
python predict.py --demo

# Single machine prediction via CLI
python predict.py --type M --air_temp 298.2 --process_temp 308.7 --speed 1400 --torque 45.0 --tool_wear 80

# Single machine under thermal load
python predict.py --type L --air_temp 302.5 --process_temp 310.2 --speed 1330 --torque 48.0 --tool_wear 110

# Interactive terminal wizard
python predict.py

# Batch fleet scoring from CSV
python predict.py --file data/ai4i2020.csv --output results/fleet_predictions.csv
```

---

## 10. Sample Prediction Diagnostic Report

```text
========================================================================
 INDUSTRIAL MACHINE PREDICTIVE MAINTENANCE DIAGNOSTIC REPORT
========================================================================
OPERATING CONDITIONS & SENSOR TELEMETRY:
  - Machine Type             : L (Low Quality Variant)
  - Air Temperature          : 302.5 K (29.4 deg C)
  - Process Temperature      : 310.2 K (37.1 deg C)
  - Rotational Speed         : 1330 RPM
  - Applied Torque           : 48.0 Nm
  - Tool Wear Duration       : 110 minutes

DERIVED PHYSICAL METRICS:
  - Temperature Differential : 7.70 K
  - Mechanical Power         : 6.69 kW
  - Overstrain Index (Wear*T): 5280.0 min*Nm

PREDICTION & HEALTH ASSESSMENT:
  - Machine Health Status    : CRITICAL (Imminent Failure Expected)
  - Failure Risk Level       : HIGH
  - Failure Likelihood       : 99.99%
  - Decision Threshold Used  : 0.53
  - Binary Prediction        : [1] FAILURE EXPECTED

FAILURE MODE DIAGNOSTICS (Statistical Likelihood Estimate):
  >>> Most Probable Mode     : Heat Dissipation Failure
  Probability Breakdown:
    * Heat Dissipation Failure  :  99.2%
    * Tool Wear Failure         :   0.8%
    * Overstrain Failure        :   0.0%
    * Power Failure             :   0.0%
  [NOTE] This diagnostic indicates statistical likelihood based on sensor telemetry,
         not a physically confirmed failure until verified by maintenance personnel.

ACTIONABLE MAINTENANCE RECOMMENDATIONS:
  CRITICAL: Heat Dissipation Failure indicated by thermal telemetry!
    - Inspect coolant fluid level, circulating pump, and cooling radiator fins.
    - Verify process-to-ambient temperature gradient (Delta-T must remain >= 8.6 K).
    - Avoid prolonged operation at low rotational speeds with high thermal load.
========================================================================
```
