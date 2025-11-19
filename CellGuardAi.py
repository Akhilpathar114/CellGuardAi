# app.py  —  CellGuard.AI Streamlit Dashboard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Helper: Simulate sample BMS data
# -----------------------------
def generate_sample_bms_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    time = np.arange(n_samples)

    voltage = 3.7 + 0.05*np.sin(time/50) + np.random.normal(0, 0.005, n_samples)
    current = 1.5 + 0.3*np.sin(time/30) + np.random.normal(0, 0.05, n_samples)
    temperature = 30 + 3*np.sin(time/60) + np.random.normal(0, 0.3, n_samples)
    soc = np.clip(80 + 10*np.sin(time/80) + np.random.normal(0, 1, n_samples), 0, 100)
    cycle = time // 50

    # Inject some stress / anomalies
    anomaly_indices = np.random.choice(n_samples, size=40, replace=False)
    voltage[anomaly_indices] -= np.random.uniform(0.05, 0.12, size=len(anomaly_indices))
    temperature[anomaly_indices] += np.random.uniform(3, 7, size=len(anomaly_indices))

    df = pd.DataFrame({
        "time": time,
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "soc": soc,
        "cycle": cycle
    })
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
    temp_threshold = df["temperature"].mean() + 2*df["temperature"].std()
    volt_drop_threshold = -0.03

    conditions = (
        (df["temperature"] > temp_threshold) |
        (df["voltage_roc"] < volt_drop_threshold)
    )
    df["risk_label"] = np.where(conditions, 1, 0)  # 1 = high risk

    return df

# -----------------------------
# 3. Train models + compute health score
# -----------------------------
def build_models_and_scores(df, contamination=0.05):
    df = df.copy()

    # Features for anomaly detection
    anomaly_features = [
        "voltage", "current", "temperature", "soc",
        "voltage_ma", "voltage_roc", "temp_roc",
        "voltage_var", "temp_ma"
    ]

    X_anomaly = df[anomaly_features]

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    iso_forest.fit(X_anomaly)
    df["anomaly_flag"] = iso_forest.predict(X_anomaly)
    df["anomaly_flag"] = df["anomaly_flag"].map({1: 0, -1: 1})

    # Decision Tree on rule-based risk label
    clf_features = anomaly_features + ["anomaly_flag"]
    X_clf = df[clf_features]
    y_clf = df["risk_label"]

    tree_clf = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )
    tree_clf.fit(X_clf, y_clf)
    df["risk_pred"] = tree_clf.predict(X_clf)

    # Create health proxy for regression
    df["health_proxy"] = (
        df["voltage_ma"].max() - df["voltage_ma"] +
        (df["temperature"] - df["temperature"].min())/10
    )

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
    score = (
        0.5 * health_component +
        0.3 * (1 - df["risk_pred"]) +
        0.2 * (1 - df["anomaly_flag"])
    )
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
def main():
    st.set_page_config(
        page_title="CellGuard.AI Dashboard",
        layout="wide"
    )

    st.title(" CellGuard.AI – Intelligent Battery Health Prediction")
    st.markdown(
        "This dashboard analyzes BMS data, detects anomalies, "
        "and computes a **Battery Health Score (0–100)** with preventive recommendations."
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")

    data_mode = st.sidebar.radio(
        "Data source:",
        ["Use built-in sample data", "Upload BMS CSV"]
    )

    contamination = st.sidebar.slider(
        "Anomaly sensitivity (Isolation Forest contamination):",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01
    )

    window = st.sidebar.slider(
        "Moving window size (for rolling features):",
        min_value=5,
        max_value=30,
        value=10,
        step=1
    )

    # Load data
    if data_mode == "Use built-in sample data":
        df_raw = generate_sample_bms_data()
        st.info("Using built-in simulated BMS-like data.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload BMS CSV file",
            type=["csv"]
        )
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            st.success("CSV file loaded successfully.")
        else:
            st.warning("Please upload a CSV file or switch to sample data.")
            st.stop()

    # Show raw data preview
    st.subheader(" Raw BMS Data (Preview)")
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
    df_out, iso_forest, tree_clf, reg = build_models_and_scores(
        df_fe,
        contamination=contamination
    )

    # Add recommendations
    df_out["recommendation"] = df_out.apply(recommend_action, axis=1)

    # ------------------ Layout: Metrics + Plots + Table ------------------
    st.subheader(" Overall Battery Health Summary")

    col1, col2, col3 = st.columns(3)

    avg_score = df_out["battery_health_score"].mean()
    high_risk_pct = (df_out["battery_health_score"] < 60).mean() * 100
    anomaly_pct = df_out["anomaly_flag"].mean() * 100

    col1.metric("Average Health Score", f"{avg_score:.1f} / 100")
    col2.metric("High-Risk Time (%)", f"{high_risk_pct:.1f}%")
    col3.metric("Anomalous Points (%)", f"{anomaly_pct:.1f}%")

    # Plots
    st.subheader(" Voltage & Anomalies Over Time")

    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(df_out["time"], df_out["voltage"], label="Voltage")
    anomaly_points = df_out[df_out["anomaly_flag"] == 1]
    ax1.scatter(
        anomaly_points["time"],
        anomaly_points["voltage"],
        marker="x",
        label="Anomaly"
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("Voltage with Detected Anomalies")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader(" Battery Health Score Over Time")

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(df_out["time"], df_out["battery_health_score"])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Health Score (0–100)")
    ax2.set_title("CellGuard.AI – Battery Health Score Timeline")
    st.pyplot(fig2)

    # Worst cases table
    st.subheader("⚠️ Most Critical Battery Moments")

    worst_n = st.slider(
        "Show lowest N health-score points:",
        min_value=5,
        max_value=50,
        value=15,
        step=5
    )

    worst_df = df_out.sort_values("battery_health_score").head(worst_n)[
        ["time", "voltage", "temperature",
         "battery_health_score", "risk_pred",
         "anomaly_flag", "recommendation"]
    ]

    st.dataframe(worst_df)

    st.markdown(
        "These are the **most stressed time-points** detected by CellGuard.AI "
        "along with suggested preventive actions."
    )

    # Allow download of processed output
    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download full processed data as CSV",
        data=csv,
        file_name="CellGuardAI_Output.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.caption("CellGuard.AI – Predicting Battery Failures Before They Start.")

if __name__ == "__main__":
    main()
