11.19 6:49 pm
# CellGuard.AI — Streamlit Dashboard v2 (Enhanced UI)

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


# -----------------------------
# 5. Streamlit UI
# -----------------------------
def add_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background: radial-gradient(circle at top left, #1f2933 0, #020617 55%);
            color: #f9fafb;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .metric-card {
            padding: 1rem 1.2rem;
            border-radius: 1rem;
            background: rgba(15,23,42,0.85);
            border: 1px solid rgba(148,163,184,0.4);
        }
        .title-gradient {
            background: linear-gradient(90deg,#38bdf8,#22c55e,#eab308);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
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
        st.markdown('<h1 class="title-gradient">CellGuard.AI</h1>', unsafe_allow_html=True)
        st.markdown(
            "AI-powered **Battery Health Intelligence** – analyze BMS data, detect hidden anomalies, "
            "and predict failures *before* they happen."
        )

    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")

    data_mode = st.sidebar.radio(
        "Data source:",
        ["Use built-in sample data", "Upload BMS CSV"],
    )

    contamination = st.sidebar.slider(
        "Anomaly sensitivity:",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help="Higher value → more points marked as anomalies.",
    )

    window = st.sidebar.slider(
        "Rolling window (feature smoothing):",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Upload CSV format** should contain at least these columns:\n"
        "`voltage, current, temperature, soc, cycle`"
    )

    # Load data
    if data_mode == "Use built-in sample data":
        df_raw = generate_sample_bms_data()
        st.info("Using built-in simulated BMS-like data.")
    else:
        uploaded_file = st.sidebar.file_uploader(" Upload BMS CSV file", type=["csv"])
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            st.success("CSV file loaded successfully.")
        else:
            st.warning("Please upload a CSV file or switch to sample data from the sidebar.")
            st.stop()

    # Show raw data preview
    with st.expander(" Raw BMS Data Preview", expanded=False):
        st.dataframe(df_raw.head())

    # Basic column check
    required_cols = ["voltage", "current", "temperature", "soc", "cycle"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        st.error(
            f"Missing required columns in data: {missing}. "
            "Please rename your columns accordingly."
        )
        st.stop()

    # Feature engineering
    df_fe = feature_engineering(df_raw, window=window)

    # Models + scores
    df_out, iso_forest, tree_clf, reg = build_models_and_scores(df_fe, contamination=contamination)

    # Add recommendations
    df_out["recommendation"] = df_out.apply(recommend_action, axis=1)

    # --------- SUMMARY METRICS ----------
    avg_score = df_out["battery_health_score"].mean()
    high_risk_pct = (df_out["battery_health_score"] < 60).mean() * 100
    anomaly_pct = df_out["anomaly_flag"].mean() * 100

    st.subheader(" Overall Battery Health Summary")

    mcol1, mcol2, mcol3, mcol4 = st.columns([1.4, 1.4, 1.4, 2])

    with mcol1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Health Score", f"{avg_score:.1f} / 100")
        st.markdown("</div>", unsafe_allow_html=True)

    with mcol2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("High-Risk Time", f"{high_risk_pct:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with mcol3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Anomalous Points", f"{anomaly_pct:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with mcol4:
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(avg_score),
                title={"text": "Pack Health Gauge"},
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
        gauge_fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(gauge_fig, use_container_width=True)

    # --------- TABS LAYOUT ----------
    tab_overview, tab_anomaly, tab_table = st.tabs(
        [" Health Timeline", " Anomaly Explorer", " Data & Download"]
    )

    # --- Tab 1: Health Timeline ---
    with tab_overview:
        st.markdown("### Battery Health Score Over Time")

        fig_health = px.area(
            df_out,
            x="time",
            y="battery_health_score",
            title="CellGuard.AI – Battery Health Score Timeline",
            labels={"time": "Time", "battery_health_score": "Health Score (0–100)"},
        )
        fig_health.update_traces(line_shape="spline")
        st.plotly_chart(fig_health, use_container_width=True)

        st.caption(
            "Higher score = healthier pack. Sudden drops can indicate stress, imbalance, or thermal events."
        )

    # --- Tab 2: Anomaly Explorer ---
    with tab_anomaly:
        st.markdown("### Voltage & Anomalies Over Time")

        anomaly_points = df_out[df_out["anomaly_flag"] == 1]

        fig_volt = px.line(
            df_out,
            x="time",
            y="voltage",
            labels={"time": "Time", "voltage": "Voltage (V)"},
        )
        fig_volt.update_traces(name="Voltage", showlegend=True)

        if not anomaly_points.empty:
            fig_volt.add_scatter(
                x=anomaly_points["time"],
                y=anomaly_points["voltage"],
                mode="markers",
                name="Anomaly",
            )

        fig_volt.update_layout(title="Voltage with Detected Anomalies")
        st.plotly_chart(fig_volt, use_container_width=True)

        st.markdown("### Most Critical Battery Moments")

        worst_n = st.slider(
            "Show lowest N health-score points:",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
        )

        worst_df = df_out.sort_values("battery_health_score").head(worst_n)[
            [
                "time",
                "voltage",
                "temperature",
                "battery_health_score",
                "risk_pred",
                "anomaly_flag",
                "recommendation",
            ]
        ]

        st.dataframe(worst_df, use_container_width=True)
        st.markdown(
            "These are the **most stressed time-points** detected by CellGuard.AI "
            "along with suggested preventive actions."
        )

    # --- Tab 3: Data & Download ---
    with tab_table:
        st.markdown("### Full Processed Dataset")
        st.dataframe(df_out.head(200), use_container_width=True)

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download full processed data as CSV",
            data=csv,
            file_name="CellGuardAI_Output.csv",
            mime="text/csv",
        )

        st.caption("Use this data for deeper analysis, model training, or reporting.")

    st.markdown("---")
    st.caption("CellGuard.AI – Predicting Battery Failures Before They Start.")


if __name__ == "__main__":
    main()
