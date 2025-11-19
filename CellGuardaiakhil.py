# CellGuard.AI — Streamlit Dashboard v3 (Enhanced UI + Info Buttons)

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# -----------------------------
# 1. Helper: Simulate sample BMS data
# -----------------------------
def generate_sample_bms_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    time = np.arange(n_samples)

    voltage = 3.7 + 0.05 * np.sin(time / 50) + np.random.normal(0, 0.005, n_samples)
    current = 1.5 + 0.3 * np.sin(time / 30) + np.random.normal(0, 0.05, n_samples)
    temperature = 30 + 3 * np.sin(time / 60) + np.random.normal(0, 0.3, n_samples)
    soc = np.clip(80 + 10 * np.sin(time / 80) + np.random.normal(0, 1, n_samples), 0, 100)
    cycle = time // 50

    # Inject some stress / anomalies
    anomaly_indices = np.random.choice(n_samples, size=40, replace=False)
    voltage[anomaly_indices] -= np.random.uniform(0.05, 0.12, size=len(anomaly_indices))
    temperature[anomaly_indices] += np.random.uniform(3, 7, size=len(anomaly_indices))

    df = pd.DataFrame(
        {
            "time": time,
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "soc": soc,
            "cycle": cycle,
        }
    )
    return df


# -----------------------------
# 2. Feature Engineering
# -----------------------------
def feature_engineering(df, window=10):
    df = df.copy()

    df["voltage_ma"] = df["voltage"].rolling(window).mean()
    df["voltage_roc"] = df["voltage"].diff()
    df["temp_roc"] = df["temperature"].diff()
    df["voltage_var"] = df["voltage"].rolling(window).var()
    df["temp_ma"] = df["temperature"].rolling(window).mean()

    # Drop NaNs from rolling
    df = df.dropna().reset_index(drop=True)

    # Create simple rule-based risk label
    temp_threshold = df["temperature"].mean() + 2 * df["temperature"].std()
    volt_drop_threshold = -0.03

    conditions = (df["temperature"] > temp_threshold) | (df["voltage_roc"] < volt_drop_threshold)
    df["risk_label"] = np.where(conditions, 1, 0)  # 1 = high risk

    return df


# -----------------------------
# 3. Train models + compute health score
# -----------------------------
def build_models_and_scores(df, contamination=0.05):
    df = df.copy()

    # Features for anomaly detection
    anomaly_features = [
        "voltage",
        "current",
        "temperature",
        "soc",
        "voltage_ma",
        "voltage_roc",
        "temp_roc",
        "voltage_var",
        "temp_ma",
    ]

    X_anomaly = df[anomaly_features]

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    iso_forest.fit(X_anomaly)
    df["anomaly_flag"] = iso_forest.predict(X_anomaly)
    df["anomaly_flag"] = df["anomaly_flag"].map({1: 0, -1: 1})

    # Decision Tree on rule-based risk label
    clf_features = anomaly_features + ["anomaly_flag"]
    X_clf = df[clf_features]
    y_clf = df["risk_label"]

    tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree_clf.fit(X_clf, y_clf)
    df["risk_pred"] = tree_clf.predict(X_clf)

    # Create health proxy for regression
    df["health_proxy"] = df["voltage_ma"].max() - df["voltage_ma"] + (df["temperature"] - df["temperature"].min()) / 10

    trend_features = ["voltage_ma", "voltage_var", "temp_ma", "cycle", "anomaly_flag"]
    X_trend = df[trend_features]
    y_trend = df["health_proxy"]

    reg = LinearRegression()
    reg.fit(X_trend, y_trend)
    df["health_pred"] = reg.predict(X_trend)

    # Normalize health_pred 0–1, invert
    hp = df["health_pred"]
    health_norm = (hp - hp.min()) / (hp.max() - hp.min() + 1e-6)
    health_component = 1 - health_norm

    # Combine into Battery Health Score
    score = 0.5 * health_component + 0.3 * (1 - df["risk_pred"]) + 0.2 * (1 - df["anomaly_flag"])
    df["battery_health_score"] = (score * 100).clip(0, 100)

    return df, iso_forest, tree_clf, reg


# -----------------------------
# 4. Recommendation logic
# -----------------------------
def recommend_action(row):
    score = row["battery_health_score"]
    if score > 85 and row["risk_pred"] == 0 and row["anomaly_flag"] == 0:
        return "Battery healthy. Normal operation."
    elif 70 < score <= 85:
        return "Monitor battery. Avoid deep discharge and frequent fast charging."
    elif 50 < score <= 70:
        return "Limit fast charging. Allow cooling intervals. Check cell balance."
    else:
        return "High risk! Reduce load, allow cooling, and schedule maintenance."


def pack_health_label(score):
    if score >= 85:
        return "HEALTHY", "#16a34a"
    elif score >= 60:
        return "WATCH", "#eab308"
    else:
        return "CRITICAL", "#dc2626"


# -----------------------------
# 5. UI helpers
# -----------------------------
def add_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background: radial-gradient(circle at top left, #020617 0, #020617 40%, #0b1120 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .glass-card {
            padding: 1.1rem 1.3rem;
            border-radius: 1.1rem;
            background: rgba(15,23,42,0.88);
            border: 1px solid rgba(148,163,184,0.35);
            box-shadow: 0 18px 45px rgba(15,23,42,0.8);
            transition: transform 0.12s ease-out, box-shadow 0.12s ease-out;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 22px 50px rgba(15,23,42,0.95);
        }
        .metric-card {
            padding: 0.9rem 1.1rem;
            border-radius: 1rem;
            background: rgba(15,23,42,0.9);
            border: 1px solid rgba(148,163,184,0.4);
        }
        .title-gradient {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            background: linear-gradient(120deg,#38bdf8,#22c55e,#eab308);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .pill-label {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .health-pill {
            background: rgba(22,163,74,0.15);
            color: #bbf7d0;
            border: 1px solid rgba(34,197,94,0.6);
        }
        .watch-pill {
            background: rgba(250,204,21,0.1);
            color: #facc15;
            border: 1px solid rgba(234,179,8,0.7);
        }
        .critical-pill {
            background: rgba(248,113,113,0.08);
            color: #fecaca;
            border: 1px solid rgba(220,38,38,0.8);
        }
        .info-badge {
            font-size: 0.8rem;
            opacity: 0.85;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="CellGuard.AI Dashboard", layout="wide")
    add_custom_css()

    # --------- HERO SECTION ----------
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.markdown("### ")
    with col_title:
        st.markdown('<div class="title-gradient">CELLGUARD.AI</div>', unsafe_allow_html=True)
        st.markdown(
            "Ultra-fast overview of **battery health, anomalies, and failure risk**. "
            "Designed for lab cells, EV packs, and future battery intelligence platforms."
        )

    st.markdown("---")

    # --------- SIDEBAR / CONFIG ----------
    st.sidebar.header("⚙️ Configuration")

    profile = st.sidebar.selectbox(
        "Application profile",
        ["Lab cell / Prototype", "2W EV (Scooter)", "3W EV (Auto / Cargo)", "4W EV / Car pack"],
    )

    data_mode = st.sidebar.radio(
        "Data source",
        ["Use built-in sample data", "Upload BMS CSV"],
    )

    # tweak default contamination by profile (small touch)
    base_contamination = {
        "Lab cell / Prototype": 0.07,
        "2W EV (Scooter)": 0.05,
        "3W EV (Auto / Cargo)": 0.04,
        "4W EV / Car pack": 0.03,
    }[profile]

    contamination = st.sidebar.slider(
        "Anomaly sensitivity",
        min_value=0.01,
        max_value=0.20,
        value=float(base_contamination),
        step=0.01,
        help="Higher value → more points marked as anomalies.",
    )

    window = st.sidebar.slider(
        "Rolling window (feature smoothing)",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### CSV format ℹ️\n"
        "- Must contain: `voltage, current, temperature, soc, cycle`\n"
        "- Optional extra columns are preserved in output."
    )

    # --------- DATA LOADING ----------
    if data_mode == "Use built-in sample data":
        df_raw = generate_sample_bms_data()
        st.info("Using built-in simulated BMS-like data (lab-style profile).")
    else:
        uploaded_file = st.sidebar.file_uploader(" Upload BMS CSV file", type=["csv"])
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            st.success("CSV file loaded successfully.")
        else:
            st.warning("Upload a CSV file or switch to sample data from the sidebar.")
            st.stop()

    with st.expander(" Raw BMS Data Preview", expanded=False):
        st.dataframe(df_raw.head(), use_container_width=True)

    required_cols = ["voltage", "current", "temperature", "soc", "cycle"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        st.error(
            f"Missing required columns: {missing}. "
            "Please add/rename these columns in your CSV."
        )
        st.stop()

    # --------- PIPELINE ----------
    df_fe = feature_engineering(df_raw, window=window)
    df_out, iso_forest, tree_clf, reg = build_models_and_scores(df_fe, contamination=contamination)
    df_out["recommendation"] = df_out.apply(recommend_action, axis=1)

    avg_score = float(df_out["battery_health_score"].mean())
    high_risk_pct = float((df_out["battery_health_score"] < 60).mean() * 100)
    anomaly_pct = float(df_out["anomaly_flag"].mean() * 100)
    label, color = pack_health_label(avg_score)

    # --------- SUMMARY CARDS ----------
    st.subheader(" Pack Health Snapshot")

    top = st.container()
    with top:
        c1, c2 = st.columns([2.1, 1.2])
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            colm1, colm2, colm3 = st.columns(3)

            with colm1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Average Health Score", f"{avg_score:.1f} / 100")
                st.markdown("</div>", unsafe_allow_html=True)

            with colm2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("High-Risk Time", f"{high_risk_pct:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)

            with colm3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Anomalous Points", f"{anomaly_pct:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("ℹ️ What do these metrics mean?", expanded=False):
                st.markdown(
                    "- **Average Health Score** – overall condition of the pack across time.\n"
                    "- **High-Risk Time** – percentage of timestamps where score < 60.\n"
                    "- **Anomalous Points** – data points where behavior is statistically unusual."
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            pill_class = (
                "health-pill" if label == "HEALTHY" else "watch-pill" if label == "WATCH" else "critical-pill"
            )
            st.markdown(
                f'<span class="pill-label {pill_class}">PACK STATUS · {label}</span>',
                unsafe_allow_html=True,
            )

            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=avg_score,
                    title={"text": "Health Gauge"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"thickness": 0.35},
                        "steps": [
                            {"range": [0, 50], "color": "#7f1d1d"},
                            {"range": [50, 80], "color": "#854d0e"},
                            {"range": [80, 100], "color": "#064e3b"},
                        ],
                    },
                )
            )
            gauge_fig.update_layout(height=210, margin=dict(l=8, r=8, t=40, b=5))
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.caption(
                "Status label summarises risk: HEALTHY (>85), WATCH (60–85), "
                "CRITICAL (<60). Use with anomaly % for decisions."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # --------- TABS ----------
    tab_overview, tab_anomaly, tab_table = st.tabs(
        [" Health Timeline", " Anomaly Explorer", " Data & Export"]
    )

    with tab_overview:
        left, right = st.columns([2.5, 1])

        with left:
            st.markdown("#### Health Score Over Time")
            fig_health = px.area(
                df_out,
                x="time",
                y="battery_health_score",
                labels={
                    "time": "Time",
                    "battery_health_score": "Health Score (0–100)",
                },
            )
            fig_health.update_traces(line_shape="spline")
            fig_health.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_health, use_container_width=True)

        with right:
            st.markdown("#### ℹ️ What this shows")
            st.markdown(
                "- Each point = one timestamp from your BMS data.\n"
                "- Smooth curve = how health changes with usage and temperature.\n"
                "- Sudden drops = stress events, imbalance, or overheating.\n"
                "- For EVs: compare before/after fast-charging or hill climbs."
            )
