"""
Predictive Maintenance & Machine Failure Detection
===================================================
Module: utils.py
Description: Shared utility module for model inference, feature engineering,
             telemetry validation, and diagnostic recommendations for the
             Streamlit web application.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

# Suppress minor version warnings
warnings.filterwarnings("ignore")


def get_base_dir():
    """Returns the base project directory."""
    return os.path.dirname(os.path.abspath(__file__))


def load_dataset(filename="ai4i2020.csv"):
    """Loads the AI4I 2020 Predictive Maintenance dataset."""
    base_dir = get_base_dir()
    possible_paths = [
        os.path.join(base_dir, "data", filename),
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "..", filename),
        os.path.join("data", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not locate dataset '{filename}' in expected paths.")


def load_binary_model():
    """
    Loads the trained production binary failure detection model bundle.
    Returns (pipeline, decision_threshold, model_name, feature_names).
    """
    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, "models", "binary_failure_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Binary model file not found at: {model_path}. Run 'python train_model.py' first.")

    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        pipeline = bundle["pipeline"]
        threshold = float(bundle.get("selected_threshold", 0.67))
        model_name = bundle.get("model_name", "Hist Gradient Boosting (Balanced)")
        features = bundle.get("feature_names", [])
    else:
        pipeline = bundle
        threshold = 0.50
        model_name = "Trained Classifier"
        features = []

    return pipeline, threshold, model_name, features


def load_failure_type_model():
    """
    Loads the secondary multiclass failure-type diagnostic pipeline.
    Returns pipeline or None if file not found.
    """
    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, "models", "failure_type_model.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def load_metrics_summary():
    """Loads saved metrics summary JSON from training evaluation."""
    base_dir = get_base_dir()
    json_path = os.path.join(base_dir, "results", "metrics_summary.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


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


def validate_telemetry(machine_type, air_temp, process_temp, speed, torque, tool_wear):
    """
    Validates sensor values against realistic physical operational ranges.
    Returns (is_valid: bool, error_msg: str, validated_dict: dict)
    """
    machine_type = str(machine_type).strip().upper()
    if machine_type not in ["L", "M", "H"]:
        return False, f"Machine type '{machine_type}' must be one of: 'L' (Low), 'M' (Medium), 'H' (High Quality Variant).", {}

    try:
        air_temp = float(air_temp)
        process_temp = float(process_temp)
        speed = float(speed)
        torque = float(torque)
        tool_wear = float(tool_wear)
    except (ValueError, TypeError) as e:
        return False, f"Telemetry readings must be numeric numbers. Detail: {e}", {}

    if air_temp < 250.0 or air_temp > 350.0:
        return False, f"Air temperature ({air_temp:.1f} K) is outside realistic factory room limits (250 K to 350 K / -23°C to 77°C).", {}

    if process_temp < 250.0 or process_temp > 370.0:
        return False, f"Process temperature ({process_temp:.1f} K) is outside realistic cutting chamber limits (250 K to 370 K).", {}

    if speed <= 0 or speed > 6000:
        return False, f"Rotational speed ({speed:.0f} RPM) must be positive and within spindle rating (1 to 6,000 RPM).", {}

    if torque < 0 or torque > 300:
        return False, f"Torque ({torque:.1f} Nm) must be non-negative and within spindle limits (0 to 300 Nm).", {}

    if tool_wear < 0 or tool_wear > 600:
        return False, f"Tool wear ({tool_wear:.0f} min) must be non-negative (0 to 600 minutes).", {}

    validated_dict = {
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }
    return True, "", validated_dict


def predict_machine_health(sensor_dict, binary_pipeline, threshold=0.67):
    """
    Runs binary failure prediction and categorizes risk.
    Returns a dictionary of structured prediction outputs.
    """
    df_raw = pd.DataFrame([sensor_dict])
    df_feat = engineer_features(df_raw)

    probabilities = binary_pipeline.predict_proba(df_feat)[0]
    fail_prob = float(probabilities[1])

    is_failure = bool(fail_prob >= threshold)
    
    if fail_prob < 0.20:
        health_status = "HEALTHY"
        status_description = "Machine is operating under optimal thermal and mechanical conditions."
        badge_color = "#10b981"  # Green
        risk_level = "LOW RISK"
    elif fail_prob < threshold:
        health_status = "WARNING"
        status_description = "Early equipment degradation or abnormal telemetry detected. Elevated risk."
        badge_color = "#f59e0b"  # Orange
        risk_level = "ELEVATED RISK"
    else:
        health_status = "CRITICAL"
        status_description = "Imminent equipment failure expected based on physical load indicators."
        badge_color = "#ef4444"  # Red
        risk_level = "HIGH RISK"

    return {
        "failure_predicted": is_failure,
        "failure_probability": fail_prob,
        "health_status": health_status,
        "risk_level": risk_level,
        "status_description": status_description,
        "badge_color": badge_color,
        "threshold_used": threshold,
        "derived_metrics": {
            "Temp_diff": float(df_feat["Temp_diff"].iloc[0]),
            "Power_kW": float(df_feat["Power_kW"].iloc[0]),
            "Tool_wear_Torque": float(df_feat["Tool_wear_Torque"].iloc[0])
        }
    }


def diagnose_failure_type(sensor_dict, type_pipeline):
    """
    Executes the secondary multiclass diagnostic pipeline.
    Returns (predicted_mode, mode_probabilities_dict).
    """
    if type_pipeline is None:
        return "Diagnostic Model Unavailable", {}

    df_raw = pd.DataFrame([sensor_dict])
    df_feat = engineer_features(df_raw)

    diag_pred = type_pipeline.predict(df_feat)[0]
    probabilities = type_pipeline.predict_proba(df_feat)[0]

    mode_probs = {
        cls: float(prob) for cls, prob in zip(type_pipeline.classes_, probabilities)
    }
    return str(diag_pred), mode_probs


def get_maintenance_recommendations(health_status, failure_type, metrics):
    """
    Generates actionable maintenance guidance based on risk level and diagnosed mode.
    """
    recs = []
    if health_status == "HEALTHY":
        recs.append("Machine telemetry is within nominal operating boundaries.")
        recs.append("Continue standard preventive maintenance schedule and continuous monitoring.")
        return recs

    if health_status == "WARNING":
        recs.append("Early warning condition: telemetry indicates elevated thermal or mechanical strain.")
        recs.append("Increase telemetry sampling frequency and log vibration/thermal trends.")
        recs.append("Schedule a visual inspection during the next routine shift changeover.")

    # Tailored recommendations based on root-cause failure mode
    if failure_type == "Heat Dissipation Failure":
        recs.append("CRITICAL: Heat Dissipation Defect Detected")
        recs.append("  - Process temperature to ambient difference is deficient (Delta-T < 8.6 K).")
        recs.append("  - Inspect coolant fluid reservoir levels and circulation pump flow rate.")
        recs.append("  - Clean cutting chamber cooling radiator fins and verify heat exchange blower operation.")
        recs.append("  - Avoid sustained low spindle speed (< 1380 RPM) under heavy cutting friction.")
    elif failure_type == "Power Failure":
        recs.append("CRITICAL: Mechanical Power / Spindle Drive Overload Detected")
        recs.append("  - Mechanical power draw (Torque × RPM) is breaching the operational envelope (3.5 - 9.0 kW).")
        recs.append("  - Check electrical drive inverter, motor cabling, and spindle bearings for drag or binding.")
        recs.append("  - Inspect workpiece alignment and cutting feed rate to prevent motor stall.")
    elif failure_type == "Overstrain Failure":
        recs.append("CRITICAL: Structural Overstrain Detected")
        recs.append("  - The product of tool wear duration and applied torque exceeds tool fracture capacity.")
        recs.append("  - Immediately decrease feed rate or reduce spindle depth of cut.")
        recs.append("  - Replace cutting blade inserts immediately to prevent catastrophic shank fracture.")
    elif failure_type == "Tool Wear Failure":
        recs.append("CRITICAL: Tool Wear Endurance Limit Exceeded")
        recs.append("  - Cumulative blade wear time has reached end-of-life threshold (200 - 240 minutes).")
        recs.append("  - Index or replace tool cutting inserts prior to commencing the next machining run.")
    else:
        recs.append("Conduct a full electromechanical diagnostic sweep across all spindle and drive assemblies.")

    return recs
