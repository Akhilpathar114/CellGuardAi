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


    df = df.dropna().reset_index(drop=True)


    temp_threshold = df["temperature"].mean() + 2 * df["temperature"].std()
    volt_drop_threshold = -0.03


    conditions = (df["temperature"] > temp_threshold) | (df["voltage_roc"] < volt_drop_threshold)
    df["risk_label"] = np.where(conditions, 1, 0)


    return df




# -----------------------------
# 3. Build Models + Health Score
# -----------------------------
def build_models_and_scores(df, contamination=0.05):
    df = df.copy()


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


    clf_features = anomaly_features + ["anomaly_flag"]
    X_clf = df[clf_features]
    y_clf = df["risk_label"]


    tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree_clf.fit(X_clf, y_clf)
    df["risk_pred"] = tree_clf.predict(X_clf)


    df["health_proxy"] = (
        df["voltage_ma"].max() - df["voltage_ma"] + (df["temperature"] - df["temperature"].min()) / 10
    )


    trend_features = ["voltage_ma", "voltage_var", "temp_ma", "cycle", "anomaly_flag"]
    X_trend = df[trend_features]
    y_trend = df["health_proxy"]


    reg = LinearRegression()
    reg.fit(X_trend, y_trend)
    df["health_pred"] = reg.predict(X_trend)


    hp = df["health_pred"]
    health_norm = (hp - hp.min()) / (hp.max() - hp.min() + 1e-6)
    health_component = 1 - health_norm


    score = (
        0.5 * health_component +
        0.3 * (1 - df["risk_pred"]) +
        0.2 * (1 - df["anomaly_flag"])
    )
    df["battery_health_score"] = (score * 100).clip(0, 100)


    return df, iso_forest, tree_clf, reg




# -----------------------------
# 4. Recommendation Logic
# -----------------------------
def recommend_action(row):
    score = row["battery_health_score"]
    if score > 85 and row["risk_pred"] == 0 and row["anomaly_flag"] == 0:
        return "Battery healthy. Normal operation."
    elif 70 < score <= 85:
        return "Monitor battery. Avoid deep discharge & fast charging."
    elif 50 < score <= 70:
        return "Limit fast charging. Allow cooling intervals."
    else:
        return "High risk! Reduce load & schedule maintenance."




def pack_health_label(score):
    if score >= 85:
        return "HEALTHY", "#16a34a"
    elif score >= 60:
        return "WATCH", "#eab308"
    else:
        return "CRITICAL", "#dc2626"
        def normalize_bms_columns(df):
            """
              Try to automatically map messy CSV column names
              to the internal names: voltage, current, temperature, soc, cycle.
            """
            df = df.copy()

            simplified = {
            col: "".join(ch for ch in col.lower() if ch.isalnum())
            for col in df.columns
    }

    patterns = {
        "voltage": ["volt", "vcell", "cellv", "packv"],
        "current": ["curr", "amp", "current", "ichg", "idis"],
        "temperature": ["temp", "temperature", "celltemp", "packtemp"],
        "soc": ["soc", "stateofcharge"],
        "cycle": ["cycle", "cyclecount", "chargecycle", "cycleindex"],
    }

    col_map = {}

    for target, keys in patterns.items():
        for orig, s in simplified.items():
            if any(k in s for k in keys):
                if target not in col_map:
                    col_map[target] = orig
                break

    rename_dict = {orig: target for target, orig in col_map.items()}
    df = df.rename(columns=rename_dict)

    return df, col_map
# -----------------------------
# 5. Custom CSS for UI
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
        </style>
        """,
        unsafe_allow_html=True,
    )




# -----------------------------
# 6. Main App
# -----------------------------
def main():
    st.set_page_config(page_title="CellGuard.AI Dashboard", layout="wide")
    add_custom_css()


    # --------- HERO SECTION ----------
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.markdown("### 🔋")
    with col_title:
        st.markdown('<div class="title-gradient">CELLGUARD.AI</div>', unsafe_allow_html=True)
        st.markdown(
            "Ultra-fast overview of **battery health, anomalies, and failure risk**. "
            "Designed for lab cells, EV packs, and future battery intelligence platforms."
        )


    st.markdown("---")


    # --------- SIDEBAR SETTINGS ----------
    st.sidebar.header("⚙️ Configuration")


    profile = st.sidebar.selectbox(
        "Application profile",
        ["Lab cell / Prototype", "2W EV (Scooter)", "3W EV (Auto / Cargo)", "4W EV / Car pack"],
    )


    data_mode = st.sidebar.radio(
        "Data source",
        ["Use built-in sample data", "Upload BMS CSV"],
    )


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
    )


    window = st.sidebar.slider(
        "Rolling window size",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
    )


    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### CSV format\n"
        "`voltage, current, temperature, soc, cycle` are required."
    )


    # --------- LOAD DATA ----------
    if data_mode == "Use built-in sample data":
        df_raw = generate_sample_bms_data()
        st.info("Using built-in simulated BMS data.")
    else:
        uploaded = st.sidebar.file_uploader("📂 Upload BMS CSV", type=["csv"])
        if uploaded is not None:
            df_raw = pd.read_csv(uploaded)
            st.success("CSV loaded successfully.")
        else:
            st.warning("Upload a CSV or switch to sample data.")
            st.stop()

    # 🔁 Try to auto-normalize messy column names
    df_raw, col_map = normalize_bms_columns(df_raw)

    if col_map:
        mapped_text = ", ".join(
            [f'"{orig}" → **{target}**' for target, orig in col_map.items()]
        )
        st.info(f"Auto-mapped columns: {mapped_text}")
    else:
        st.warning(
            "Could not auto-detect standard columns. "
            "Make sure voltage/current/temperature/SOC/cycle columns exist in your file."
        )

    with st.expander("👀 Raw BMS Preview", expanded=False):
        st.dataframe(df_raw.head(), use_container_width=True)

    required_cols = ["voltage", "current", "temperature", "soc", "cycle"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        st.error(
            f"Still missing required logical columns: {missing}. "
            "Please rename your columns or adjust the CSV."
        )
        st.stop()
    


    # --------- PIPELINE ----------
    df_fe = feature_engineering(df_raw, window)
    df_out, iso_forest, tree_clf, reg = build_models_and_scores(df_fe, contamination)
    df_out["recommendation"] = df_out.apply(recommend_action, axis=1)


    avg_score = float(df_out["battery_health_score"].mean())
    high_risk_pct = float((df_out["battery_health_score"] < 60).mean() * 100)
    anomaly_pct = float(df_out["anomaly_flag"].mean() * 100)


    label, color = pack_health_label(avg_score)


    # -----------------------------
    # 7. SUMMARY CARDS
    # -----------------------------
    st.subheader("📊 Pack Health Summary")


    top_c1, top_c2 = st.columns([2.1, 1.2])


    # --- CARD 1: Metrics ---
    with top_c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)


        with m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Health Score", f"{avg_score:.1f} / 100")
            st.markdown("</div>", unsafe_allow_html=True)


        with m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("High-Risk Time", f"{high_risk_pct:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)


        with m3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Anomalies %", f"{anomaly_pct:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)


        with st.expander("ℹ️ About these metrics"):
            st.markdown(
                "- **Avg Score**: Overall pack condition.\n"
                "- **High-Risk Time**: % of readings < 60 health.\n"
                "- **Anomalies %**: ML-detected unusual patterns."
            )
        st.markdown("</div>", unsafe_allow_html=True)


    # --- CARD 2: GAUGE ---
    with top_c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)


        pill_class = (
            "health-pill" if label == "HEALTHY"
            else "watch-pill" if label == "WATCH"
            else "critical-pill"
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


        st.caption("HEALTHY >85 • WATCH 60–85 • CRITICAL <60")
        st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("---")
# -----------------------------
    # 8. TABS SECTION
    # -----------------------------
    tab_overview, tab_anomaly, tab_table = st.tabs(
        ["📈 Health Timeline", "🚨 Anomaly Explorer", "📄 Data & Export"]
    )


    # -----------------------------
    # TAB 1: HEALTH TIMELINE
    # -----------------------------
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
                "- Each point = one reading from your BMS data.\n"
                "- Curve shows overall battery health evolution.\n"
                "- Sudden drops = thermal stress / imbalance / voltage sag.\n"
                "- Compare pre/post charging, hill climbs, or heavy loads."
            )


    # -----------------------------
    # TAB 2: ANOMALY EXPLORER
    # -----------------------------
    with tab_anomaly:
        st.markdown("#### Voltage & Anomaly Points")


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
                marker=dict(color="red", size=8, symbol="x"),
            )


        fig_volt.update_layout(
            title="Voltage with ML-Detected Anomalies",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_volt, use_container_width=True)


        with st.expander("ℹ️ How anomalies are found"):
            st.markdown(
                "- We use **Isolation Forest**, an unsupervised ML algorithm.\n"
                "- Flags unusual voltage/current/temp patterns.\n"
                "- Great for early detection of failing cells."
            )


        st.markdown("#### Most Critical Battery Moments")


        worst_n = st.slider(
            "Show worst N health points",
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
        st.caption("These timestamps show highest stress & lowest health.")


    # -----------------------------
    # TAB 3: DATA & EXPORT
    # -----------------------------
    with tab_table:
        st.markdown("#### Processed Data")
        st.dataframe(df_out.head(300), use_container_width=True)


        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download full processed data as CSV",
            data=csv,
            file_name="CellGuardAI_Output.csv",
            mime="text/csv",
        )


        with st.expander("ℹ️ Tips for using this CSV"):
            st.markdown(
                "- Use `battery_health_score` for forecasting.\n"
                "- `anomaly_flag` + `risk_pred` for classification ML.\n"
                "- Rolling features help in training deep learning models.\n"
                "- Ideal for EV fleet analytics & long-term pack health monitoring."
            )


    st.markdown("---")
    st.caption("CellGuard.AI — Predicting battery failures before they start.")




# ENTRYPOINT
if __name__ == "__main__":

    main()

