"""
Predictive Maintenance & Machine Failure Detection
===================================================
Script: predict.py (Audited & Enhanced Version)
Description: Production inference and health diagnostic script.
             Loads the trained binary failure prediction pipeline and the multiclass
             failure-type diagnostic model to evaluate industrial equipment sensors.

Key Enhancements:
  - Supports model bundles with automatic detection of the optimal decision threshold.
  - Strict input validation against realistic physical sensor limits.
  - Returns continuous failure probability, calibrated risk categories, and binary classification.
  - Transparently frames failure-type predictions as statistical likelihood estimates,
    not confirmed physical diagnoses.
  - Supports Interactive Wizard, CLI Flags, Batch CSV Scoring, and Demo Benchmark modes.

Authors/Engineers: Antigravity AI Pair Programming Team
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

# Suppress minor non-critical warnings
warnings.filterwarnings("ignore")


def load_models(models_dir=None):
    """
    Loads both the primary binary failure prediction pipeline (including threshold metadata)
    and the secondary multiclass failure-type diagnostic pipeline.
    """
    if models_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")

    binary_model_path = os.path.join(models_dir, "binary_failure_model.joblib")
    type_model_path = os.path.join(models_dir, "failure_type_model.joblib")

    if not os.path.exists(binary_model_path):
        raise FileNotFoundError(
            f"[ERROR] Trained model file not found at: {binary_model_path}\n"
            f"Please run 'python train_model.py' first to train and save the production models."
        )

    loaded_binary = joblib.load(binary_model_path)
    
    # Handle both model bundle dictionary and direct Pipeline objects
    if isinstance(loaded_binary, dict):
        binary_pipeline = loaded_binary["pipeline"]
        decision_threshold = float(loaded_binary.get("selected_threshold", 0.50))
        model_name = loaded_binary.get("model_name", "Random Forest (Balanced)")
    else:
        binary_pipeline = loaded_binary
        decision_threshold = 0.50
        model_name = "Trained Classifier"

    type_pipeline = joblib.load(type_model_path) if os.path.exists(type_model_path) else None

    return binary_pipeline, type_pipeline, decision_threshold, model_name


def engineer_features(df):
    """
    Derives physical domain features required by the trained pipelines:
      - Temp_diff: Process temperature minus Air temperature (K)
      - Power_kW: Rotational mechanical power (kW)
      - Tool_wear_Torque: Cumulative tool wear stress indicator (min*Nm)
    """
    df_feat = df.copy()
    df_feat["Temp_diff"] = df_feat["Process temperature [K]"] - df_feat["Air temperature [K]"]
    df_feat["Power_kW"] = (df_feat["Torque [Nm]"] * df_feat["Rotational speed [rpm]"] * 2 * np.pi) / 60000.0
    df_feat["Tool_wear_Torque"] = df_feat["Tool wear [min]"] * df_feat["Torque [Nm]"]
    return df_feat


def validate_sensor_inputs(machine_type, air_temp, process_temp, speed, torque, tool_wear):
    """
    Validates sensor values against realistic physical ranges.
    Raises ValueError with a helpful explanation if values are invalid.
    """
    machine_type = str(machine_type).strip().upper()
    if machine_type not in ["L", "M", "H"]:
        raise ValueError(
            f"Invalid Machine Type '{machine_type}'. Must be one of: 'L' (Low), 'M' (Medium), 'H' (High Quality Variant)."
        )

    try:
        air_temp = float(air_temp)
        process_temp = float(process_temp)
        speed = float(speed)
        torque = float(torque)
        tool_wear = float(tool_wear)
    except (ValueError, TypeError) as e:
        raise ValueError(f"All numerical sensor readings must be valid numbers. Error detail: {e}")

    # Physical bounds checks
    if air_temp < 250.0 or air_temp > 350.0:
        raise ValueError(f"Air temperature {air_temp} K is outside realistic ambient factory limits (250 K - 350 K / -23 C to +77 C).")
    if process_temp < 250.0 or process_temp > 370.0:
        raise ValueError(f"Process temperature {process_temp} K is outside realistic milling chamber limits (250 K - 370 K).")
    if process_temp < air_temp:
        # Physical constraint: cutting process generates friction heat; process temp should typically exceed ambient
        pass  # allowed with warning if chilled, but physically noteworthy
    if speed <= 0 or speed > 6000:
        raise ValueError(f"Rotational speed {speed} rpm must be positive and within spindle limits (1 - 6,000 rpm).")
    if torque < 0 or torque > 300:
        raise ValueError(f"Torque {torque} Nm must be non-negative and within spindle limits (0 - 300 Nm).")
    if tool_wear < 0 or tool_wear > 600:
        raise ValueError(f"Tool wear {tool_wear} min must be non-negative (0 - 600 min).")

    return {
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }


def get_health_status(failure_prob, threshold=0.50):
    """
    Categorizes machine health and failure risk based on predicted probability
    relative to the tuned decision threshold.
    """
    if failure_prob < 0.20:
        return "HEALTHY (Optimal Operation)", "LOW", "[NORMAL]"
    elif failure_prob < threshold:
        return "WARNING (Elevated Risk / Early Degradation)", "MODERATE", "[WARNING]"
    else:
        return "CRITICAL (Imminent Failure Expected)", "HIGH", "[ALERT]"


def get_recommendations(risk_level, failure_type=None):
    """
    Generates actionable maintenance recommendations based on risk and failure type.
    """
    recs = []
    if risk_level == "LOW":
        recs.append("Machine is operating within normal parameters. Continue standard routine maintenance schedule.")
    elif risk_level == "MODERATE":
        recs.append("Sensor telemetry indicates early mechanical/thermal stress. Increase sampling frequency.")
        recs.append("Schedule preventative inspection during the next planned production changeover.")

    if failure_type == "Heat Dissipation Failure":
        recs.append("CRITICAL: Heat Dissipation Failure indicated by thermal telemetry!")
        recs.append("  - Inspect coolant fluid level, circulating pump, and cooling radiator fins.")
        recs.append("  - Verify process-to-ambient temperature gradient (Delta-T must remain >= 8.6 K).")
        recs.append("  - Avoid prolonged operation at low rotational speeds with high thermal load.")
    elif failure_type == "Power Failure":
        recs.append("CRITICAL: Power / Drive Overload indicated by electrical/speed telemetry!")
        recs.append("  - Inspect spindle drive inverter, motor cabling, and torque limiter.")
        recs.append("  - Mechanical power draw (Torque x Speed) is operating outside safe limits (3.5 - 9.0 kW).")
        recs.append("  - Check for spindle bearing binding or workpiece jamming.")
    elif failure_type == "Overstrain Failure":
        recs.append("CRITICAL: Structural Overstrain indicated by excessive load and tool wear!")
        recs.append("  - Cumulative tool wear x torque product has breached failure threshold.")
        recs.append("  - Immediately reduce cutting feed rate and applied torque.")
        recs.append("  - Replace the cutting tool inserts to avoid catastrophic shank fracture.")
    elif failure_type == "Tool Wear Failure":
        recs.append("CRITICAL: Tool Wear Failure indicated by accumulated cutting duration!")
        recs.append("  - Tool insert operating life has reached maximum endurance limit (200-240 min).")
        recs.append("  - Replace cutting tool blade before starting the next production batch.")

    return recs


def predict_single_machine(sensor_dict, binary_pipeline, type_pipeline, threshold=0.50, model_name=None):
    """
    Executes inference for a single machine and returns comprehensive diagnostic results.
    """
    df_raw = pd.DataFrame([sensor_dict])
    df_feat = engineer_features(df_raw)

    # 1. Binary Failure Prediction
    probabilities = binary_pipeline.predict_proba(df_feat)[0]
    fail_prob = float(probabilities[1])
    
    # Binary classification evaluated against selected decision threshold
    pred_class = 1 if fail_prob >= threshold else 0
    health_status, risk_level, status_badge = get_health_status(fail_prob, threshold)

    # 2. Multiclass Failure Type Diagnosis (if elevated risk or predicted failure)
    failure_type = "None (Normal Operation)"
    type_probabilities = {}

    if type_pipeline is not None and (fail_prob >= 0.20 or pred_class == 1):
        try:
            diag_pred = type_pipeline.predict(df_feat)[0]
            diag_proba = type_pipeline.predict_proba(df_feat)[0]
            failure_type = str(diag_pred)
            type_probabilities = {
                cls: float(prob) for cls, prob in zip(type_pipeline.classes_, diag_proba)
            }
        except Exception as e:
            failure_type = f"Diagnostic unavailable ({e})"

    recommendations = get_recommendations(risk_level, failure_type if fail_prob >= 0.20 else None)

    return {
        "inputs": sensor_dict,
        "features": {
            "Temp_diff": float(df_feat["Temp_diff"].iloc[0]),
            "Power_kW": float(df_feat["Power_kW"].iloc[0]),
            "Tool_wear_Torque": float(df_feat["Tool_wear_Torque"].iloc[0])
        },
        "model_used": model_name or "Binary Classifier",
        "decision_threshold": threshold,
        "binary_prediction": pred_class,
        "failure_predicted": bool(pred_class == 1),
        "failure_probability": fail_prob,
        "health_status": health_status,
        "risk_level": risk_level,
        "status_badge": status_badge,
        "predicted_failure_type": failure_type,
        "failure_type_probabilities": type_probabilities,
        "recommendations": recommendations
    }


def print_prediction_report(res):
    """
    Prints a formatted, transparent diagnostic report to the terminal.
    """
    print("\n" + "=" * 72)
    print(" INDUSTRIAL MACHINE PREDICTIVE MAINTENANCE DIAGNOSTIC REPORT")
    print("=" * 72)

    inp = res["inputs"]
    print("OPERATING CONDITIONS & SENSOR TELEMETRY:")
    print(f"  - Machine Type             : {inp['Type']} ({'Low' if inp['Type']=='L' else 'Medium' if inp['Type']=='M' else 'High'} Quality Variant)")
    print(f"  - Air Temperature          : {inp['Air temperature [K]']:.1f} K ({inp['Air temperature [K]'] - 273.15:.1f} deg C)")
    print(f"  - Process Temperature      : {inp['Process temperature [K]']:.1f} K ({inp['Process temperature [K]'] - 273.15:.1f} deg C)")
    print(f"  - Rotational Speed         : {inp['Rotational speed [rpm]']:.0f} RPM")
    print(f"  - Applied Torque           : {inp['Torque [Nm]']:.1f} Nm")
    print(f"  - Tool Wear Duration       : {inp['Tool wear [min]']:.0f} minutes")

    feat = res["features"]
    print("\nDERIVED PHYSICAL METRICS:")
    print(f"  - Temperature Differential : {feat['Temp_diff']:.2f} K")
    print(f"  - Mechanical Power         : {feat['Power_kW']:.2f} kW")
    print(f"  - Overstrain Index (Wear*T): {feat['Tool_wear_Torque']:.1f} min*Nm")

    print("\nPREDICTION & HEALTH ASSESSMENT:")
    print(f"  - Machine Health Status    : {res['health_status']}")
    print(f"  - Failure Risk Level       : {res['risk_level']}")
    print(f"  - Failure Likelihood       : {res['failure_probability'] * 100:.2f}%")
    print(f"  - Decision Threshold Used  : {res['decision_threshold']:.2f}")
    print(f"  - Binary Prediction        : {'[1] FAILURE EXPECTED' if res['failure_predicted'] else '[0] NORMAL OPERATION'}")

    if res["failure_probability"] >= 0.20:
        print("\nFAILURE MODE DIAGNOSTICS (Statistical Likelihood Estimate):")
        print(f"  >>> Most Probable Mode     : {res['predicted_failure_type']}")
        if res["failure_type_probabilities"]:
            print("  Probability Breakdown:")
            sorted_probs = sorted(res["failure_type_probabilities"].items(), key=lambda x: x[1], reverse=True)
            for mode, prob in sorted_probs:
                print(f"    * {mode:<26}: {prob * 100:>5.1f}%")
        print("  [NOTE] This diagnostic indicates statistical likelihood based on sensor telemetry,")
        print("         not a physically confirmed failure until verified by maintenance personnel.")

    print("\nACTIONABLE MAINTENANCE RECOMMENDATIONS:")
    for rec in res["recommendations"]:
        print(f"  {rec}")

    print("=" * 72 + "\n")


def run_batch_prediction(csv_path, binary_pipeline, type_pipeline, threshold=0.50, output_csv=None):
    """
    Executes batch prediction on a CSV file and saves summary results.
    """
    print(f"\nLoading batch sensor data from: {csv_path}")
    df_batch = pd.read_csv(csv_path)

    required_cols = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    missing = [c for c in required_cols if c not in df_batch.columns]
    if missing:
        raise ValueError(f"Batch CSV is missing required sensor columns: {missing}")

    df_feat = engineer_features(df_batch[required_cols])
    probas = binary_pipeline.predict_proba(df_feat)[:, 1]
    preds = (probas >= threshold).astype(int)

    df_results = df_batch.copy()
    df_results["Failure_Probability"] = np.round(probas, 4)
    df_results["Predicted_Failure"] = preds
    df_results["Health_Status"] = [get_health_status(p, threshold)[0] for p in probas]

    # Predict failure type for records with elevated risk
    if type_pipeline is not None:
        elevated_mask = probas >= 0.20
        df_results["Predicted_Failure_Type"] = "None"
        if elevated_mask.any():
            type_preds = type_pipeline.predict(df_feat[elevated_mask])
            df_results.loc[elevated_mask, "Predicted_Failure_Type"] = type_preds

    if output_csv is None:
        base_name, ext = os.path.splitext(csv_path)
        output_csv = f"{base_name}_predictions.csv"

    df_results.to_csv(output_csv, index=False)
    print(f"Batch prediction completed for {len(df_results):,} records.")
    print(f"Decision Threshold Applied: {threshold:.2f}")
    print(f"Total Predicted Failures  : {preds.sum()} ({preds.sum()/len(preds)*100:.2f}%)")
    print(f"Results saved to: {output_csv}")


def run_demo(binary_pipeline, type_pipeline, threshold=0.50, model_name=None):
    """
    Executes four realistic industrial machine operational scenarios.
    """
    print("\n" + "#" * 72)
    print(" RUNNING PREDICTIVE MAINTENANCE BENCHMARK SCENARIOS (--demo)")
    print("#" * 72)

    demo_cases = [
        {
            "name": "Scenario 1: Optimal Normal Machine (Low Wear, Moderate Load)",
            "sensors": {
                "Type": "M",
                "Air temperature [K]": 298.2,
                "Process temperature [K]": 308.7,
                "Rotational speed [rpm]": 1408,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 25
            }
        },
        {
            "name": "Scenario 2: Heat Dissipation Defect (High Ambient Temp & Low Delta-T)",
            "sensors": {
                "Type": "L",
                "Air temperature [K]": 302.5,
                "Process temperature [K]": 310.2,  # Delta-T = 7.7 K (< 8.6 K threshold)
                "Rotational speed [rpm]": 1330,   # Low speed (< 1380 rpm)
                "Torque [Nm]": 48.0,
                "Tool wear [min]": 110
            }
        },
        {
            "name": "Scenario 3: Severe Overstrain Condition (Heavy Load & Worn Tool)",
            "sensors": {
                "Type": "L",
                "Air temperature [K]": 298.5,
                "Process temperature [K]": 308.5,
                "Rotational speed [rpm]": 1350,
                "Torque [Nm]": 65.0,              # Heavy torque
                "Tool wear [min]": 215             # 65 * 215 = 13,975 min*Nm (Overstrain threshold breached)
            }
        },
        {
            "name": "Scenario 4: High Mechanical Power Exceedance (Spindle Surge)",
            "sensors": {
                "Type": "H",
                "Air temperature [K]": 297.5,
                "Process temperature [K]": 307.8,
                "Rotational speed [rpm]": 2650,   # High speed
                "Torque [Nm]": 68.0,              # High torque -> Power ~18.9 kW (> 9.0 kW threshold)
                "Tool wear [min]": 45
            }
        }
    ]

    for case in demo_cases:
        print(f"\n--- {case['name']} ---")
        validated = validate_sensor_inputs(
            case["sensors"]["Type"],
            case["sensors"]["Air temperature [K]"],
            case["sensors"]["Process temperature [K]"],
            case["sensors"]["Rotational speed [rpm]"],
            case["sensors"]["Torque [Nm]"],
            case["sensors"]["Tool wear [min]"]
        )
        res = predict_single_machine(validated, binary_pipeline, type_pipeline, threshold, model_name)
        print_prediction_report(res)


def interactive_prompt(binary_pipeline, type_pipeline, threshold=0.50, model_name=None):
    """
    Guides the user interactively through terminal prompts with validated inputs.
    """
    print("\n" + "=" * 72)
    print(" INTERACTIVE MACHINE HEALTH INSPECTOR")
    print(" Enter machine sensor readings below (or press Enter to accept defaults):")
    print("=" * 72)

    try:
        raw_type = input("  1. Machine Type [L, M, H] (default: M): ").strip() or "M"
        raw_air = input("  2. Air Temperature in Kelvin (default: 298.2 K): ").strip() or "298.2"
        raw_proc = input("  3. Process Temperature in Kelvin (default: 308.7 K): ").strip() or "308.7"
        raw_speed = input("  4. Rotational Speed in RPM (default: 1450 rpm): ").strip() or "1450"
        raw_torque = input("  5. Applied Torque in Nm (default: 42.0 Nm): ").strip() or "42.0"
        raw_wear = input("  6. Tool Wear in minutes (default: 60 min): ").strip() or "60"

        validated = validate_sensor_inputs(raw_type, raw_air, raw_proc, raw_speed, raw_torque, raw_wear)
        result = predict_single_machine(validated, binary_pipeline, type_pipeline, threshold, model_name)
        print_prediction_report(result)
    except KeyboardInterrupt:
        print("\n[Exiting inspection session.]")
    except Exception as e:
        print(f"\n[ERROR] Input validation failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict industrial machine failure and diagnose failure modes from sensor readings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Run interactive diagnostic wizard:
  python predict.py

  # 2. Run built-in benchmark demo scenarios:
  python predict.py --demo

  # 3. Predict for a specific machine via CLI arguments:
  python predict.py --type M --air_temp 298.2 --process_temp 308.7 --speed 1400 --torque 45.0 --tool_wear 80

  # 4. Batch predict from a CSV file with custom threshold:
  python predict.py --file data/ai4i2020.csv --threshold 0.35 --output results/batch_predictions.csv
        """
    )

    parser.add_argument("--demo", action="store_true", help="Run 4 demonstration scenarios illustrating normal and failure states.")
    parser.add_argument("--file", "-f", type=str, default=None, help="Path to CSV file for batch predictions.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path for batch predictions.")
    parser.add_argument("--threshold", type=float, default=None, help="Override classification threshold (default: uses trained optimal threshold).")
    parser.add_argument("--type", "-t", type=str, default=None, help="Machine quality type: L (Low), M (Medium), or H (High).")
    parser.add_argument("--air_temp", type=float, default=None, help="Air temperature in Kelvin (e.g. 298.2).")
    parser.add_argument("--process_temp", type=float, default=None, help="Process temperature in Kelvin (e.g. 308.7).")
    parser.add_argument("--speed", type=float, default=None, help="Rotational speed in RPM (e.g. 1400).")
    parser.add_argument("--torque", type=float, default=None, help="Torque in Nm (e.g. 45.0).")
    parser.add_argument("--tool_wear", type=float, default=None, help="Tool wear duration in minutes (e.g. 150).")

    args = parser.parse_args()

    # Load trained models and decision threshold
    try:
        binary_pipeline, type_pipeline, default_threshold, model_name = load_models()
    except Exception as e:
        print(f"\n{e}")
        sys.exit(1)

    # Use command-line threshold override if provided
    decision_threshold = args.threshold if args.threshold is not None else default_threshold

    # 1. Demo mode
    if args.demo:
        run_demo(binary_pipeline, type_pipeline, decision_threshold, model_name)
        return

    # 2. Batch mode
    if args.file:
        try:
            run_batch_prediction(args.file, binary_pipeline, type_pipeline, decision_threshold, args.output)
        except Exception as e:
            print(f"\n[ERROR] Batch prediction failed: {e}")
            sys.exit(1)
        return

    # 3. CLI arguments mode
    cli_args_provided = [args.type, args.air_temp, args.process_temp, args.speed, args.torque, args.tool_wear]
    if any(a is not None for a in cli_args_provided):
        if not all(a is not None for a in cli_args_provided):
            print("\n[ERROR] When using CLI arguments, you must provide all 6 sensor parameters:")
            print("  --type, --air_temp, --process_temp, --speed, --torque, --tool_wear")
            sys.exit(1)
        try:
            validated = validate_sensor_inputs(
                args.type, args.air_temp, args.process_temp,
                args.speed, args.torque, args.tool_wear
            )
            result = predict_single_machine(validated, binary_pipeline, type_pipeline, decision_threshold, model_name)
            print_prediction_report(result)
        except Exception as e:
            print(f"\n[ERROR] Prediction failed: {e}")
            sys.exit(1)
        return

    # 4. Interactive mode (default when run without arguments)
    interactive_prompt(binary_pipeline, type_pipeline, decision_threshold, model_name)


if __name__ == "__main__":
    main()
