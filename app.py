"""
Predictive Maintenance & Machine Failure Detection System
==========================================================
Interactive Web Application Dashboard (Streamlit & Plotly)
Authors/Engineers: Senior Full-Stack & ML Engineering Team
Dataset: UCI AI4I 2020 Predictive Maintenance Dataset
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import shared inference and utility functions
import utils

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM INDUSTRIAL THEME CSS
# ==============================================================================
st.set_page_config(
    page_title="Previsio — AI Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Modern Industrial High-Contrast Theme
CUSTOM_CSS = """
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Custom Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e3a8a 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        border: 1px solid #1e3a8a;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
    }
    .header-banner h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff !important;
    }
    .header-banner p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #94a3b8 !important;
    }

    /* Metric Cards */
    .metric-card {
        background: #111827 !important;
        padding: 18px 20px;
        border-radius: 12px;
        border: 1px solid #1f2937 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.5);
        border-color: #374151 !important;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8 !important;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc !important;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 11px;
        color: #64748b !important;
        margin-top: 4px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .status-healthy {
        background-color: rgba(34, 197, 94, 0.15) !important;
        color: #4ade80 !important;
        border: 1px solid rgba(74, 222, 128, 0.4) !important;
    }
    .status-warning {
        background-color: rgba(245, 158, 11, 0.15) !important;
        color: #fbbf24 !important;
        border: 1px solid rgba(251, 191, 36, 0.4) !important;
    }
    .status-critical {
        background-color: rgba(239, 68, 68, 0.15) !important;
        color: #f87171 !important;
        border: 1px solid rgba(248, 113, 113, 0.4) !important;
    }

    /* Recommendation Alert Boxes */
    .rec-box {
        background-color: #111827 !important;
        border-left: 4px solid #3b82f6 !important;
        padding: 18px 22px;
        border-radius: 10px;
        border-top: 1px solid #1f2937 !important;
        border-right: 1px solid #1f2937 !important;
        border-bottom: 1px solid #1f2937 !important;
        margin-top: 16px;
    }
    .rec-box h4 {
        margin: 0 0 10px 0;
        font-size: 15px;
        font-weight: 600;
        color: #f8fafc !important;
    }
    .rec-box ul {
        margin: 0;
        padding-left: 20px;
        color: #cbd5e1 !important;
        font-size: 13.5px;
        line-height: 1.7;
    }

    /* Technical Callout Card */
    .tech-card {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .tech-card h3 {
        margin-top: 0;
        font-size: 17px;
        font-weight: 600;
        color: #f8fafc !important;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-size: 14px !important;
        padding: 6px 12px;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background-color: #1e293b !important;
    }

    /* Form Container */
    [data-testid="stForm"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
        padding: 24px !important;
    }

    /* Input Widget Labels (CRITICAL: 100% VISIBLE) */
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    .stSelectbox label,
    .stNumberInput label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    /* Subtext and Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
        font-size: 12px !important;
    }

    /* Input Fields (Selectbox and NumberInput) */
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] div {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="input"] button {
        background-color: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
    }
    div[data-baseweb="input"] button:hover {
        color: #ffffff !important;
        background-color: #334155 !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }

    /* Preset & Secondary Buttons */
    div.stButton > button {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25) !important;
    }
    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    div.stButton > button:hover {
        background-color: #2563eb !important;
        border-color: #60a5fa !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px);
    }
    div.stButton > button:hover p,
    div.stButton > button:hover span,
    div.stButton > button:hover div {
        color: #ffffff !important;
    }

    /* Form Submit Button (Primary Action) */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid #60a5fa !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stFormSubmitButton"] > button span {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
        transform: translateY(-1px);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. CACHED MODEL & DATASET LOADERS
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_cached_models():
    """Caches model pipelines in memory to prevent reloading on every run."""
    try:
        bin_pipe, threshold, model_name, features = utils.load_binary_model()
    except Exception as e:
        bin_pipe, threshold, model_name, features = None, 0.67, "Unavailable", []
    
    try:
        type_pipe = utils.load_failure_type_model()
    except Exception:
        type_pipe = None

    return bin_pipe, threshold, model_name, features, type_pipe


@st.cache_data(show_spinner=False)
def get_cached_dataset():
    """Caches the dataset in memory."""
    try:
        return utils.load_dataset()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_cached_metrics():
    """Caches training evaluation metrics summary."""
    return utils.load_metrics_summary()


# Initialize cached components & dynamic evaluation metrics
binary_pipeline, default_threshold, model_name, feature_names, type_pipeline = get_cached_models()
df_telemetry = get_cached_dataset()
metrics_eval = utils.get_evaluated_metrics()
prod_metrics = metrics_eval["production"]
base_metrics = metrics_eval["baseline"]
diag_accuracy = metrics_eval["diagnostic_accuracy"]
all_test_metrics = metrics_eval["all_test_metrics"]


# ==============================================================================
# 3. SIDEBAR NAVIGATION & SYSTEM HEALTH BADGE
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 20px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">
                ⚙️ Previsio
            </div>
            <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px;">
                Industrial Health & Predictive Maintenance
            </div>
        </div>
    """, unsafe_allow_html=True)

    # System Status Pill
    if binary_pipeline is not None:
        st.markdown("""
            <div style="background-color: #134e4a; border: 1px solid #2dd4bf; border-radius: 8px; padding: 10px 14px; margin-bottom: 24px;">
                <div style="font-size: 11px; color: #99f6e4; font-weight: 700; text-transform: uppercase;">System Status</div>
                <div style="font-size: 13px; color: #ffffff; font-weight: 600; margin-top: 2px;">🟢 Production Models Online</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background-color: #7f1d1d; border: 1px solid #f87171; border-radius: 8px; padding: 10px 14px; margin-bottom: 24px;">
                <div style="font-size: 11px; color: #fecaca; font-weight: 700; text-transform: uppercase;">System Status</div>
                <div style="font-size: 13px; color: #ffffff; font-weight: 600; margin-top: 2px;">🔴 Model Checkpoint Missing</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 11px; font-weight: 700; text-transform: uppercase; color: #64748b;'>Navigation</p>", unsafe_allow_html=True)
    nav_choice = st.radio(
        label="Select Workspace",
        options=[
            "📊 Dashboard Overview",
            "🔍 Machine Health Prediction",
            "🩺 Failure-Type Diagnosis",
            "📈 Telemetry Analytics"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 12px; color: #94a3b8; line-height: 1.5;">
            <strong>Dataset:</strong> AI4I 2020 Predictive Maintenance<br>
            <strong>UCI Repository ID:</strong> 601<br>
            <strong>Telemetry Fleet:</strong> 10,000 Machines
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 4. VIEW 1: DASHBOARD OVERVIEW
# ==============================================================================
if nav_choice == "📊 Dashboard Overview":
    st.markdown("""
        <div class="header-banner">
            <h1>Previsio: Industrial Equipment Health & Maintenance Center</h1>
            <p>Real-time machine telemetry monitoring, failure risk scoring, and multiclass root-cause diagnostics.</p>
        </div>
    """, unsafe_allow_html=True)

    # Quick Top Metrics
    total_machines = len(df_telemetry) if not df_telemetry.empty else 10000
    failure_machines = int(df_telemetry["Machine failure"].sum()) if not df_telemetry.empty else 339
    healthy_machines = total_machines - failure_machines

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Analyzed Fleet Size</div>
                <div class="metric-value">{total_machines:,}</div>
                <div class="metric-sub">Operational industrial records</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Healthy Machines</div>
                <div class="metric-value" style="color: #4ade80;">{healthy_machines:,}</div>
                <div class="metric-sub">{healthy_machines/total_machines*100:.2f}% normal operating condition</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Machines At Risk</div>
                <div class="metric-value" style="color: #f87171;">{failure_machines:,}</div>
                <div class="metric-sub">{failure_machines/total_machines*100:.2f}% incident rate (1 : 28.5)</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Failure Recall Rate</div>
                <div class="metric-value" style="color: #60a5fa;">{prod_metrics['recall'] * 100:.2f}%</div>
                <div class="metric-sub">{prod_metrics['tp']} of {prod_metrics['tp'] + prod_metrics['fn']} test failures detected early</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Model Performance: Reported Baseline vs. Independently Verified Production
    st.markdown("### 🏆 Model Performance Benchmark: Reported Baseline vs. Verified Production")
    st.caption("Note: Offline test evaluation metrics reflect statistical discriminatory power, not a guarantee of live zero-downtime.")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"""
            <div class="tech-card">
                <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Baseline Candidate Model</div>
                <h3 style="margin-top: 4px;">{base_metrics['model_name']}</h3>
                <p style="font-size: 13px; color: #94a3b8; margin-bottom: 16px;">
                    Standard reference model evaluated on 20% test partition (n=2,000).
                </p>
                <table style="width: 100%; font-size: 13.5px; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #1f2937; height: 32px;">
                        <td style="color: #94a3b8;">Overall Test Accuracy</td>
                        <td style="font-weight: 700; text-align: right; color: #f8fafc;">{base_metrics['accuracy'] * 100:.2f}%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1f2937; height: 32px;">
                        <td style="color: #94a3b8;">Failure-Class Precision</td>
                        <td style="font-weight: 700; text-align: right; color: #f8fafc;">{base_metrics['precision'] * 100:.2f}%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1f2937; height: 32px;">
                        <td style="color: #94a3b8;">Failure-Class Recall</td>
                        <td style="font-weight: 700; text-align: right; color: #f8fafc;">{base_metrics['recall'] * 100:.2f}%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1f2937; height: 32px;">
                        <td style="color: #94a3b8;">Failure-Class F1-Score</td>
                        <td style="font-weight: 700; text-align: right; color: #f8fafc;">{base_metrics['f1_score']:.4f}</td>
                    </tr>
                    <tr style="height: 32px;">
                        <td style="color: #94a3b8;">Area Under ROC Curve</td>
                        <td style="font-weight: 700; text-align: right; color: #f8fafc;">{base_metrics['roc_auc']:.4f}</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
            <div class="tech-card" style="border: 2px solid #3b82f6;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; font-weight: 700; color: #60a5fa; text-transform: uppercase;">Active Production Model</span>
                    <span style="background-color: #1e3a8a; color: #93c5fd; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px;">VERIFIED</span>
                </div>
                <h3 style="margin-top: 4px;">Hist Gradient Boosting (Optimal Threshold: {prod_metrics['threshold']:.2f})</h3>
                <p style="font-size: 13px; color: #94a3b8; margin-bottom: 16px;">
                    Audited with 5-Fold Stratified Cross-Validation + PR-AUC Rare Event Tuning.
                </p>
                <table style="width: 100%; font-size: 13.5px; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #1f2937; height: 32px;">
                        <td style="color: #94a3b8;">Verified Test Accuracy</td>
                        <td style="font-weight: 700; text-align: right; color: #4ade80;">{prod_metrics['accuracy'] * 100:.2f}%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1f2937; height: 32px;">
                        <td style="color: #94a3b8;">Verified Failure Precision</td>
                        <td style="font-weight: 700; text-align: right; color: #4ade80;">{prod_metrics['precision'] * 100:.2f}% (Only {prod_metrics['fp']} False Alarms)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1f2937; height: 32px;">
                        <td style="color: #94a3b8;">Verified Failure Recall</td>
                        <td style="font-weight: 700; text-align: right; color: #4ade80;">{prod_metrics['recall'] * 100:.2f}% (Catches {prod_metrics['tp']}/{prod_metrics['tp'] + prod_metrics['fn']} Failures)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1f2937; height: 32px;">
                        <td style="color: #94a3b8;">Failure-Class F1-Score</td>
                        <td style="font-weight: 700; text-align: right; color: #4ade80;">{prod_metrics['f1_score']:.4f}</td>
                    </tr>
                    <tr style="height: 32px;">
                        <td style="color: #94a3b8;">Diagnostic Multiclass Accuracy</td>
                        <td style="font-weight: 700; text-align: right; color: #60a5fa;">{diag_accuracy * 100:.2f}%</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)

    # Interactive Overview Charts
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### 🎯 Fleet Operating Condition Distribution")
        pie_fig = go.Figure(data=[go.Pie(
            labels=["Normal Healthy (0)", "Machine Failure (1)"],
            values=[healthy_machines, failure_machines],
            hole=0.6,
            marker=dict(colors=["#10b981", "#ef4444"], line=dict(color="#111827", width=2)),
            textinfo="label+percent",
            hoverinfo="label+value"
        )])
        pie_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            showlegend=False
        )
        st.plotly_chart(pie_fig, width="stretch")

    with g2:
        st.markdown("#### 🔍 Breakdown of Known Failure Modes")
        mode_counts = {"Heat Dissipation": 115, "Power Failure": 95, "Overstrain": 98, "Tool Wear": 46}
        bar_fig = go.Figure(data=[go.Bar(
            x=list(mode_counts.keys()),
            y=list(mode_counts.values()),
            marker_color=["#ef4444", "#3b82f6", "#8b5cf6", "#f59e0b"],
            text=list(mode_counts.values()),
            textposition="auto"
        )])
        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            xaxis_title="Failure Mode",
            yaxis_title="Incident Count"
        )
        st.plotly_chart(bar_fig, width="stretch")


# ==============================================================================
# 5. VIEW 2: MACHINE HEALTH PREDICTION
# ==============================================================================
elif nav_choice == "🔍 Machine Health Prediction":
    st.markdown("""
        <div class="header-banner">
            <h1>Single Equipment Health & Failure Assessment</h1>
            <p>Input real-time sensor telemetry to estimate failure likelihood and receive prescriptive maintenance guidance.</p>
        </div>
    """, unsafe_allow_html=True)

    if binary_pipeline is None:
        st.error("Binary failure detection model not found. Please train models using 'python train_model.py'.")
        st.stop()

    # Preset Scenarios for demonstration
    st.markdown("##### ⚡ Quick Demonstration Presets:")
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)

    # Session state initialization for presets
    if "p_type" not in st.session_state:
        st.session_state.p_type = "M"
        st.session_state.p_air = 298.2
        st.session_state.p_proc = 308.7
        st.session_state.p_speed = 1408
        st.session_state.p_torque = 40.0
        st.session_state.p_wear = 25

    if col_pre1.button("🟢 Normal Machine"):
        st.session_state.p_type, st.session_state.p_air, st.session_state.p_proc = "M", 298.2, 308.7
        st.session_state.p_speed, st.session_state.p_torque, st.session_state.p_wear = 1408, 40.0, 25

    if col_pre2.button("🔴 Heat Dissipation Failure"):
        st.session_state.p_type, st.session_state.p_air, st.session_state.p_proc = "L", 302.5, 310.2
        st.session_state.p_speed, st.session_state.p_torque, st.session_state.p_wear = 1330, 48.0, 110

    if col_pre3.button("🔴 Overstrain Condition"):
        st.session_state.p_type, st.session_state.p_air, st.session_state.p_proc = "L", 298.5, 308.5
        st.session_state.p_speed, st.session_state.p_torque, st.session_state.p_wear = 1350, 65.0, 215

    if col_pre4.button("🔴 Spindle Power Surge"):
        st.session_state.p_type, st.session_state.p_air, st.session_state.p_proc = "H", 297.5, 307.8
        st.session_state.p_speed, st.session_state.p_torque, st.session_state.p_wear = 2650, 68.0, 45

    st.write("")

    # Form with input controls
    with st.form("machine_telemetry_form"):
        st.markdown("### 📋 Enter Operating Sensor Telemetry")
        
        f1, f2, f3 = st.columns(3)
        with f1:
            m_type = st.selectbox(
                "Machine Variant / Quality Tier",
                options=["L", "M", "H"],
                index=["L", "M", "H"].index(st.session_state.p_type),
                help="L = Low (50% of units), M = Medium (30%), H = High Quality Variant (20%)."
            )
            air_temp = st.number_input(
                "Air Temperature [K]",
                min_value=250.0, max_value=350.0,
                value=float(st.session_state.p_air),
                step=0.1,
                help="Ambient temperature in Kelvin (298.15 K = 25.0 °C)."
            )
            st.caption(f"Equivalent: {air_temp - 273.15:.1f} °C")

        with f2:
            proc_temp = st.number_input(
                "Process Temperature [K]",
                min_value=250.0, max_value=370.0,
                value=float(st.session_state.p_proc),
                step=0.1,
                help="Cutting chamber process temperature in Kelvin."
            )
            st.caption(f"Equivalent: {proc_temp - 273.15:.1f} °C")

            rot_speed = st.number_input(
                "Rotational Speed [RPM]",
                min_value=1, max_value=6000,
                value=int(st.session_state.p_speed),
                step=10,
                help="Spindle rotational velocity in revolutions per minute."
            )

        with f3:
            torque = st.number_input(
                "Applied Torque [Nm]",
                min_value=0.0, max_value=300.0,
                value=float(st.session_state.p_torque),
                step=0.5,
                help="Torque load applied by spindle in Newton-meters."
            )

            tool_wear = st.number_input(
                "Tool Wear Duration [min]",
                min_value=0, max_value=600,
                value=int(st.session_state.p_wear),
                step=5,
                help="Cumulative active cutting duration for the blade insert in minutes."
            )

        submit_btn = st.form_submit_button("⚡ Analyze Machine Health", use_container_width=True)

    if submit_btn:
        is_valid, err_msg, sdict = utils.validate_telemetry(m_type, air_temp, proc_temp, rot_speed, torque, tool_wear)
        if not is_valid:
            st.error(f"Input Validation Error: {err_msg}")
        else:
            # Perform inference
            result = utils.predict_machine_health(sdict, binary_pipeline, threshold=default_threshold)
            fail_prob = result["failure_probability"]
            health_status = result["health_status"]
            badge_color = result["badge_color"]
            derived = result["derived_metrics"]

            st.write("---")
            st.markdown("### 📊 Diagnostic Evaluation Results")

            # Main Result Banner
            res_c1, res_c2 = st.columns([1.2, 1])
            with res_c1:
                badge_class = "status-healthy" if health_status == "HEALTHY" else "status-warning" if health_status == "WARNING" else "status-critical"
                st.markdown(f"""
                    <div style="background: #111827; padding: 24px; border-radius: 12px; border: 1px solid #1f2937; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                        <span class="status-badge {badge_class}">{health_status} CONDITION</span>
                        <h2 style="margin: 12px 0 6px 0; font-size: 26px; color: #f8fafc;">
                            {'Failure Expected' if result['failure_predicted'] else 'Normal Operation'}
                        </h2>
                        <p style="font-size: 14px; color: #94a3b8; margin-bottom: 16px;">
                            {result['status_description']}
                        </p>
                        <div style="font-size: 13px; color: #64748b;">
                            <strong>Evaluated Decision Threshold:</strong> {default_threshold:.2f} (Probabilities ≥ {default_threshold:.2f} trigger proactive alerts)
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with res_c2:
                # Gauge Meter for Failure Probability
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fail_prob * 100,
                    number={"suffix": "%", "valueformat": ".1f"},
                    title={"text": "Estimated Failure Likelihood", "font": {"size": 14, "color": "#f8fafc"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                        "bar": {"color": badge_color},
                        "steps": [
                            {"range": [0, 20], "color": "rgba(34, 197, 94, 0.25)"},
                            {"range": [20, default_threshold * 100], "color": "rgba(245, 158, 11, 0.25)"},
                            {"range": [default_threshold * 100, 100], "color": "rgba(239, 68, 68, 0.25)"}
                        ],
                        "threshold": {
                            "line": {"color": "#ef4444", "width": 3},
                            "thickness": 0.75,
                            "value": default_threshold * 100
                        }
                    }
                ))
                gauge_fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=210,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(gauge_fig, width="stretch")

            # Physical Engineering Indicators
            st.markdown("#### 🔬 Physics-Governed Operational Indicators")
            m1, m2, m3 = st.columns(3)
            with m1:
                dt_val = derived["Temp_diff"]
                dt_alert = dt_val < 8.6
                st.metric(
                    label="Thermal Differential (ΔT)",
                    value=f"{dt_val:.2f} K",
                    delta="- Low Heat Transfer Warning" if dt_alert else "Sufficient Dissipation",
                    delta_color="inverse" if dt_alert else "normal",
                    help="Process temp minus ambient air temp. Heat dissipation fails when ΔT < 8.6 K at low speed."
                )
            with m2:
                pw_val = derived["Power_kW"]
                pw_alert = pw_val < 3.5 or pw_val > 9.0
                st.metric(
                    label="Mechanical Power Draw",
                    value=f"{pw_val:.2f} kW",
                    delta="Exceeds 3.5 - 9.0 kW Envelope" if pw_alert else "Within Power Envelope",
                    delta_color="inverse" if pw_alert else "normal",
                    help="Calculated as (Torque × RPM × 2π) / 60,000. Power failures occur outside safe operating envelope."
                )
            with m3:
                os_val = derived["Tool_wear_Torque"]
                os_alert = os_val > 11000.0
                st.metric(
                    label="Tool Overstrain Index",
                    value=f"{os_val:.1f} min·Nm",
                    delta="Overstrain Risk High" if os_alert else "Normal Mechanical Load",
                    delta_color="inverse" if os_alert else "normal",
                    help="Product of tool wear time and torque load. High index risks catastrophic insert fracture."
                )

            # Actionable Maintenance Recommendations
            predicted_mode = "None"
            if health_status != "HEALTHY" and type_pipeline is not None:
                predicted_mode, _ = utils.diagnose_failure_type(sdict, type_pipeline)

            recommendations = utils.get_maintenance_recommendations(health_status, predicted_mode, derived)

            st.markdown(f"""
                <div class="rec-box">
                    <h4>🛠️ Prescriptive Engineering & Maintenance Guidance:</h4>
                    <ul>
                        {''.join(f'<li>{r}</li>' for r in recommendations)}
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    # Batch Fleet Telemetry Prediction & CSV Export
    st.write("")
    with st.expander("📁 Batch Fleet Telemetry Prediction & CSV Export", expanded=False):
        st.markdown("#### Upload Fleet Sensor Telemetry File")
        st.markdown(
            "Upload an industrial telemetry CSV file containing columns: `Type`, `Air temperature [K]`, "
            "`Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`."
        )

        col_up1, col_up2 = st.columns([3, 1])
        with col_up1:
            uploaded_batch_file = st.file_uploader(
                "Select CSV File",
                type=["csv"],
                help="Upload CSV formatted telemetry to batch predict failure likelihood for an entire equipment fleet."
            )
        with col_up2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            # Sample CSV download button
            sample_template = pd.DataFrame([
                {"Type": "M", "Air temperature [K]": 298.2, "Process temperature [K]": 308.7, "Rotational speed [rpm]": 1408, "Torque [Nm]": 40.0, "Tool wear [min]": 25},
                {"Type": "L", "Air temperature [K]": 302.5, "Process temperature [K]": 310.2, "Rotational speed [rpm]": 1330, "Torque [Nm]": 48.0, "Tool wear [min]": 110},
                {"Type": "L", "Air temperature [K]": 298.5, "Process temperature [K]": 308.5, "Rotational speed [rpm]": 1350, "Torque [Nm]": 65.0, "Tool wear [min]": 215},
                {"Type": "H", "Air temperature [K]": 297.5, "Process temperature [K]": 307.8, "Rotational speed [rpm]": 2650, "Torque [Nm]": 68.0, "Tool wear [min]": 45}
            ])
            st.download_button(
                label="📄 Download Sample CSV",
                data=sample_template.to_csv(index=False),
                file_name="sample_telemetry_template.csv",
                mime="text/csv",
                use_container_width=True
            )

        if uploaded_batch_file is not None:
            try:
                raw_upload_df = pd.read_csv(uploaded_batch_file)
                is_batch_valid, batch_err, cleaned_batch_df = utils.validate_telemetry_dataframe(raw_upload_df)
                if not is_batch_valid:
                    st.error(f"Batch Validation Error: {batch_err}")
                else:
                    with st.spinner("Scoring fleet telemetry with production models..."):
                        req_cols = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
                        df_feat_batch = utils.engineer_features(cleaned_batch_df[req_cols])
                        probas_batch = binary_pipeline.predict_proba(df_feat_batch)[:, 1]
                        preds_batch = (probas_batch >= default_threshold).astype(int)

                        scored_df = cleaned_batch_df.copy()
                        scored_df["Failure_Probability"] = np.round(probas_batch, 4)
                        scored_df["Predicted_Failure"] = preds_batch
                        scored_df["Health_Status"] = [
                            "HEALTHY" if p < 0.20 else "WARNING" if p < default_threshold else "CRITICAL"
                            for p in probas_batch
                        ]

                        if type_pipeline is not None:
                            elev_mask = probas_batch >= 0.20
                            scored_df["Predicted_Failure_Type"] = "None"
                            if elev_mask.any():
                                scored_df.loc[elev_mask, "Predicted_Failure_Type"] = type_pipeline.predict(df_feat_batch[elev_mask])

                    # Summary cards
                    total_records = len(scored_df)
                    failure_count = int(preds_batch.sum())
                    healthy_count = total_records - failure_count

                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.metric("Total Machines Scored", f"{total_records:,}")
                    with sc2:
                        st.metric("Normal Machines", f"{healthy_count:,}", f"{healthy_count/total_records*100:.1f}%")
                    with sc3:
                        st.metric("Machines At Risk", f"{failure_count:,}", f"{failure_count/total_records*100:.1f}%", delta_color="inverse")

                    # Preview Table
                    st.dataframe(scored_df.head(100), use_container_width=True)
                    if total_records > 100:
                        st.caption(f"Showing first 100 of {total_records:,} records.")

                    # Download button
                    csv_export = scored_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Full Scored Predictions (CSV)",
                        data=csv_export,
                        file_name="previsio_fleet_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error processing uploaded CSV: {e}")


# ==============================================================================
# 6. VIEW 3: FAILURE-TYPE DIAGNOSIS
# ==============================================================================
elif nav_choice == "🩺 Failure-Type Diagnosis":
    st.markdown("""
        <div class="header-banner">
            <h1>Multiclass Failure Mechanism Diagnostic Center</h1>
            <p>Diagnose the underlying root-cause failure mode when equipment exhibits abnormal mechanical or thermal strain.</p>
        </div>
    """, unsafe_allow_html=True)

    if type_pipeline is None:
        st.warning("Multiclass failure-type model not found at 'models/failure_type_model.joblib'.")
        st.stop()

    st.markdown("""
        > [!NOTE]
        > **Audit & Evaluation Scope**: This diagnostic model is trained strictly on verified machine failure instances ($n=330$) 
        > across the 4 verified physical modes: **Heat Dissipation Failure (HDF)**, **Power Failure (PWF)**, 
        > **Overstrain Failure (OSF)**, and **Tool Wear Failure (TWF)**. Random Failure (RNF) was audited and confirmed to occur 
        > in only 1 genuine failure case across 10,000 records. Predictions represent statistical likelihood suggestions, not confirmed physical diagnoses.
    """)

    with st.expander("📝 Enter or Adjust Diagnostic Sensor Inputs", expanded=True):
        d1, d2, d3 = st.columns(3)
        with d1:
            diag_type = st.selectbox("Machine Type", options=["L", "M", "H"], index=0)
            diag_air = st.number_input("Air Temp [K]", 250.0, 350.0, 302.5, 0.1)
        with d2:
            diag_proc = st.number_input("Process Temp [K]", 250.0, 370.0, 310.2, 0.1)
            diag_speed = st.number_input("Rotational Speed [RPM]", 1, 6000, 1330, 10)
        with d3:
            diag_torque = st.number_input("Torque [Nm]", 0.0, 300.0, 48.0, 0.5)
            diag_wear = st.number_input("Tool Wear [min]", 0, 600, 110, 5)

        run_diag_btn = st.button("🩺 Run Root-Cause Diagnosis", use_container_width=True)

    if run_diag_btn:
        is_v, msg, diag_sdict = utils.validate_telemetry(diag_type, diag_air, diag_proc, diag_speed, diag_torque, diag_wear)
        if not is_v:
            st.error(msg)
        else:
            predicted_mode, mode_probs = utils.diagnose_failure_type(diag_sdict, type_pipeline)

            st.write("---")
            c_diag1, c_diag2 = st.columns([1.1, 1.2])

            with c_diag1:
                st.markdown(f"""
                    <div class="tech-card" style="border-left: 5px solid #3b82f6;">
                        <div style="font-size: 11px; font-weight: 700; color: #60a5fa; text-transform: uppercase;">Diagnostic Output</div>
                        <h2 style="margin: 8px 0; color: #f8fafc; font-size: 22px;">{predicted_mode}</h2>
                        <p style="font-size: 13.5px; color: #94a3b8;">
                            The telemetry signature aligns most strongly with <strong>{predicted_mode}</strong> physics.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # Physical cause summary
                st.markdown("#### 📖 Physical Failure Mechanism:")
                if predicted_mode == "Heat Dissipation Failure":
                    st.info("Triggered when process-to-ambient temperature gradient ΔT is below 8.6 K at rotational speeds below 1380 RPM, causing thermal breakdown.")
                elif predicted_mode == "Power Failure":
                    st.info("Triggered when rotational mechanical power (Torque × Speed) falls below 3.5 kW (stall/jamming) or exceeds 9.0 kW (motor overload).")
                elif predicted_mode == "Overstrain Failure":
                    st.info("Triggered when the product of cutting tool wear time and applied torque exceeds tool endurance limits, risking structural blade chipping.")
                elif predicted_mode == "Tool Wear Failure":
                    st.info("Triggered when cumulative tool insert operating duration reaches blade replacement limits (200 - 240 minutes).")

            with c_diag2:
                st.markdown("#### 📊 Failure Mode Class Probabilities")
                if mode_probs:
                    prob_df = pd.DataFrame({
                        "Failure Mode": list(mode_probs.keys()),
                        "Probability": [v * 100 for v in mode_probs.values()]
                    }).sort_values(by="Probability", ascending=True)

                    bar_diag = px.bar(
                        prob_df,
                        x="Probability",
                        y="Failure Mode",
                        orientation="h",
                        text=prob_df["Probability"].apply(lambda x: f"{x:.1f}%"),
                        color="Probability",
                        color_continuous_scale="Blues",
                        range_x=[0, 105]
                    )
                    bar_diag.update_layout(
                        template="plotly_dark",
                        height=260,
                        margin=dict(l=20, r=20, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                        xaxis_title="Confidence Probability (%)",
                        yaxis_title=""
                    )
                    st.plotly_chart(bar_diag, width="stretch")


# ==============================================================================
# 7. VIEW 4: TELEMETRY ANALYTICS
# ==============================================================================
elif nav_choice == "📈 Telemetry Analytics":
    st.markdown("""
        <div class="header-banner">
            <h1>Interactive Sensor Telemetry & Evaluation Analytics</h1>
            <p>Explore fleet sensor distributions, feature correlations, and verified machine learning evaluation matrices.</p>
        </div>
    """, unsafe_allow_html=True)

    if df_telemetry.empty:
        st.warning("Dataset file 'ai4i2020.csv' could not be loaded for telemetry analytics.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔬 Sensor Distributions & Scatter", "🔗 Feature Correlations", "🎯 Model Evaluation Metrics"])

    with tab1:
        st.markdown("### 🔭 Bivariate Sensor Telemetry Scatter")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            x_axis = st.selectbox("X-Axis Telemetry", ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]", "Air temperature [K]"], index=0)
        with sc2:
            y_axis = st.selectbox("Y-Axis Telemetry", ["Rotational speed [rpm]", "Torque [Nm]", "Process temperature [K]", "Tool wear [min]"], index=0)
        with sc3:
            color_by = st.selectbox("Color Dimension", ["Machine failure", "Type"], index=0)

        scatter_fig = px.scatter(
            df_telemetry.sample(min(3000, len(df_telemetry)), random_state=42),
            x=x_axis,
            y=y_axis,
            color=color_by,
            color_discrete_map={0: "#3b82f6", 1: "#ef4444", "L": "#3b82f6", "M": "#10b981", "H": "#f59e0b"},
            opacity=0.65,
            hover_data=["Product ID", "Machine failure"]
        )
        scatter_fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(scatter_fig, width="stretch")

    with tab2:
        st.markdown("### 🧬 Sensor Telemetry Pearson Correlation Matrix")
        numeric_cols = [
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
            "Machine failure"
        ]
        corr = df_telemetry[numeric_cols].corr()

        heatmap_fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto"
        )
        heatmap_fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(heatmap_fig, width="stretch")

    with tab3:
        st.markdown("### 📊 Verified Confusion Matrix (Untouched Test Split: 2,000 samples)")
        # Real measured confusion matrix values loaded dynamically
        cm_data = [[prod_metrics["tn"], prod_metrics["fp"]], [prod_metrics["fn"], prod_metrics["tp"]]]
        cm_fig = px.imshow(
            cm_data,
            labels=dict(x="Predicted Condition", y="True Condition", color="Count"),
            x=["Predicted Normal (0)", "Predicted Failure (1)"],
            y=["Actual Normal (0)", "Actual Failure (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        cm_fig.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(cm_fig, width="stretch")

        st.caption(
            f"Measured at optimal threshold {prod_metrics['threshold']:.2f}: "
            f"True Negatives = {prod_metrics['tn']:,} | False Positives = {prod_metrics['fp']:,} | "
            f"False Negatives = {prod_metrics['fn']:,} | True Positives = {prod_metrics['tp']:,}"
        )

        if all_test_metrics:
            st.markdown("#### 📋 Comprehensive Model Benchmark Comparison (Test Split: 2,000 samples)")
            comp_display = []
            for m in all_test_metrics:
                comp_display.append({
                    "Model Architecture": m.get("Model", ""),
                    "Threshold": f"{m.get('Threshold', 0.5):.2f}",
                    "Accuracy": f"{m.get('Accuracy', 0) * 100:.2f}%",
                    "Precision": f"{m.get('Precision', 0) * 100:.2f}%",
                    "Recall": f"{m.get('Recall', 0) * 100:.2f}%",
                    "F1-Score": f"{m.get('F1_Score', 0):.4f}",
                    "ROC-AUC": f"{m.get('ROC_AUC', 0):.4f}",
                    "PR-AUC": f"{m.get('PR_AUC', 0):.4f}"
                })
            st.dataframe(pd.DataFrame(comp_display), use_container_width=True)



