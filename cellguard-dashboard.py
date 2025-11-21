# CellGuard.AI — Streamlit Dashboard v4 (Clean UI + Smart Features)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

# -----------------------------
# 0. Enhanced Smart Column Mapping
# -----------------------------
def normalize_bms_columns(df):
    """Smart column detection with fuzzy matching and confidence scores."""
    df = df.copy()
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Create simplified versions for matching
    simplified = {
        col: "".join(ch for ch in col.lower() if ch.isalnum())
        for col in df.columns
    }
    
    # Extended pattern matching with priority order
    patterns = {
        "voltage": ["voltage", "volt", "vcell", "cellv", "packv", "vbat", "cellvoltage", "v1", "vpack"],
        "current": ["current", "curr", "amp", "amps", "ichg", "idis", "ibat", "chargecurrent", "i1"],
        "temperature": ["temperature", "temp", "celltemp", "packtemp", "tbat", "thermaltemp", "t1"],
        "soc": ["soc", "stateofcharge", "charge", "batterylevel", "level", "capacity", "remaining"],
        "cycle": ["cycle", "cyclecount", "chargecycle", "cycleindex", "cycles", "ncycle"],
    }
    
    col_map = {}
    confidence = {}
    
    for target, keys in patterns.items():
        best_match = None
        best_score = 0
        
        for orig, s in simplified.items():
            for i, k in enumerate(keys):
                if k in s or s in k:
                    score = len(k) / len(s) if len(s) > 0 else 0
                    priority_boost = (len(keys) - i) / len(keys)
                    final_score = score * 0.7 + priority_boost * 0.3
                    
                    if final_score > best_score and orig not in col_map.values():
                        best_score = final_score
                        best_match = orig
            
            # Exact match bonus
            if s == target:
                best_match = orig
                best_score = 1.0
                break
        
        if best_match:
            col_map[target] = best_match
            confidence[target] = min(best_score, 1.0)
    
    # Auto-detect numeric columns for unmapped required fields
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    unmapped_numeric = [c for c in numeric_cols if c not in col_map.values()]
    
    # Try to fill missing required columns based on value ranges
    required = ["voltage", "current", "temperature", "soc", "cycle"]
    for req in required:
        if req not in col_map and unmapped_numeric:
            for col in unmapped_numeric:
                vals = df[col].dropna()
                if len(vals) == 0:
                    continue
                mean_val = vals.mean()
                
                # Heuristic detection based on typical value ranges
                if req == "voltage" and 2.5 <= mean_val <= 4.5:
                    col_map[req] = col
                    confidence[req] = 0.5
                    unmapped_numeric.remove(col)
                    break
                elif req == "temperature" and 15 <= mean_val <= 60:
                    col_map[req] = col
                    confidence[req] = 0.5
                    unmapped_numeric.remove(col)
                    break
                elif req == "soc" and 0 <= mean_val <= 100:
                    col_map[req] = col
                    confidence[req] = 0.5
                    unmapped_numeric.remove(col)
                    break
    
    rename_dict = {orig: target for target, orig in col_map.items()}
    df = df.rename(columns=rename_dict)
    
    return df, col_map, confidence


def generate_sample_bms_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    time = np.arange(n_samples)
    
    voltage = 3.7 + 0.05 * np.sin(time / 50) + np.random.normal(0, 0.005, n_samples)
    current = 1.5 + 0.3 * np.sin(time / 30) + np.random.normal(0, 0.05, n_samples)
    temperature = 30 + 3 * np.sin(time / 60) + np.random.normal(0, 0.3, n_samples)
    soc = np.clip(80 + 10 * np.sin(time / 80) + np.random.normal(0, 1, n_samples), 0, 100)
    cycle = time // 50
    
    # Inject realistic anomalies
    anomaly_indices = np.random.choice(n_samples, size=40, replace=False)
    voltage[anomaly_indices] -= np.random.uniform(0.05, 0.12, size=len(anomaly_indices))
    temperature[anomaly_indices] += np.random.uniform(3, 7, size=len(anomaly_indices))
    
    return pd.DataFrame({
        "time": time, "voltage": voltage, "current": current,
        "temperature": temperature, "soc": soc, "cycle": cycle,
    })


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


def build_models_and_scores(df, contamination=0.05):
    df = df.copy()
    
    features = ["voltage", "current", "temperature", "soc", "voltage_ma", 
                "voltage_roc", "temp_roc", "voltage_var", "temp_ma"]
    X = df[features]
    
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso.fit(X)
    df["anomaly_flag"] = (iso.predict(X) == -1).astype(int)
    
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(df[features + ["anomaly_flag"]], df["risk_label"])
    df["risk_pred"] = tree.predict(df[features + ["anomaly_flag"]])
    
    df["health_proxy"] = (df["voltage_ma"].max() - df["voltage_ma"] + 
                          (df["temperature"] - df["temperature"].min()) / 10)
    
    reg = LinearRegression()
    reg.fit(df[["voltage_ma", "voltage_var", "temp_ma", "cycle", "anomaly_flag"]], df["health_proxy"])
    df["health_pred"] = reg.predict(df[["voltage_ma", "voltage_var", "temp_ma", "cycle", "anomaly_flag"]])
    
    hp = df["health_pred"]
    health_norm = (hp - hp.min()) / (hp.max() - hp.min() + 1e-6)
    score = 0.5 * (1 - health_norm) + 0.3 * (1 - df["risk_pred"]) + 0.2 * (1 - df["anomaly_flag"])
    df["battery_health_score"] = (score * 100).clip(0, 100)
    
    return df


def get_recommendation(score, risk, anomaly):
    if score > 85 and risk == 0 and anomaly == 0:
        return "✅ Healthy", "Normal operation recommended"
    elif score > 70:
        return "⚠️ Monitor", "Avoid deep discharge & fast charging"
    elif score > 50:
        return "🔶 Caution", "Limit fast charging, allow cooling"
    else:
        return "🔴 Critical", "Reduce load & schedule maintenance"


def get_status(score):
    if score >= 85:
        return "HEALTHY", "#10b981", "🟢"
    elif score >= 60:
        return "WATCH", "#f59e0b", "🟡"
    else:
        return "CRITICAL", "#ef4444", "🔴"


# -----------------------------
# Custom CSS - Clean & Modern
# -----------------------------
def add_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    .block-container { padding: 1.5rem 2rem; max-width: 1300px; }
    
    .header-container {
        display: flex; align-items: center; gap: 1rem;
        padding: 1rem 0; margin-bottom: 1rem;
    }
    .logo { font-size: 2.5rem; }
    .brand { font-size: 1.8rem; font-weight: 700; 
        background: linear-gradient(90deg, #3b82f6, #10b981);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .tagline { color: #94a3b8; font-size: 0.9rem; margin-top: 0.2rem; }
    
    .card {
        background: rgba(30, 41, 59, 0.8); border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 1.2rem; margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .metric-box {
        background: rgba(15, 23, 42, 0.6); border-radius: 10px;
        padding: 1rem; text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; 
        letter-spacing: 0.05em; margin-top: 0.3rem; }
    
    .status-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.4rem 1rem; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
    }
    .status-healthy { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .status-watch { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .status-critical { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    
    .mapping-item {
        display: inline-flex; align-items: center; gap: 0.5rem;
        background: rgba(59, 130, 246, 0.1); border-radius: 6px;
        padding: 0.3rem 0.6rem; margin: 0.2rem; font-size: 0.8rem;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .mapping-original { color: #94a3b8; }
    .mapping-arrow { color: #3b82f6; }
    .mapping-target { color: #60a5fa; font-weight: 600; }
    .mapping-confidence { color: #10b981; font-size: 0.7rem; }
    
    .tip-box {
        background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        font-size: 0.85rem; color: #cbd5e1;
    }
    
    div[data-testid="stMetric"] { background: transparent; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { 
        background: rgba(30, 41, 59, 0.6); border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem; color: #94a3b8;
    }
    .stTabs [aria-selected="true"] { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# Main Application
# -----------------------------
def main():
    st.set_page_config(page_title="CellGuard.AI", page_icon="🔋", layout="wide")
    add_styles()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="logo">🔋</div>
        <div>
            <div class="brand">CellGuard.AI</div>
            <div class="tagline">Battery Health Intelligence • Anomaly Detection • Risk Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        profile = st.selectbox("Vehicle Profile", 
            ["Lab Cell / Prototype", "2W EV (Scooter)", "3W EV (Auto)", "4W EV / Car"])
        
        contamination_defaults = {"Lab Cell / Prototype": 0.07, "2W EV (Scooter)": 0.05, 
                                   "3W EV (Auto)": 0.04, "4W EV / Car": 0.03}
        
        st.markdown("---")
        st.markdown("### 📊 Data Source")
        
        data_mode = st.radio("Select source", ["Sample Data", "Upload CSV"], label_visibility="collapsed")
        
        uploaded = None
        if data_mode == "Upload CSV":
            uploaded = st.file_uploader("Drop your BMS CSV here", type=["csv"], 
                                        help="Columns: voltage, current, temperature, soc, cycle")
        
        st.markdown("---")
        st.markdown("### 🎛️ Analysis Tuning")
        
        contamination = st.slider("Anomaly Sensitivity", 0.01, 0.20, 
                                   contamination_defaults[profile], 0.01,
                                   help="Higher = more sensitive to anomalies")
        
        window = st.slider("Smoothing Window", 5, 30, 10, 
                          help="Rolling average window size")
    
    # Load Data
    if data_mode == "Sample Data":
        df_raw = generate_sample_bms_data()
        st.info("📊 Using simulated BMS data for demonstration")
    elif uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_raw):,} rows from your CSV")
    else:
        st.warning("👆 Please upload a CSV file or switch to Sample Data")
        st.stop()
    
    # Smart Column Mapping
    df_raw, col_map, confidence = normalize_bms_columns(df_raw)
    
    if col_map:
        st.markdown("#### 🔄 Smart Column Mapping")
        mapping_html = ""
        for target, orig in col_map.items():
            conf = confidence.get(target, 0)
            conf_text = f"{conf*100:.0f}%" if conf < 1 else "exact"
            mapping_html += f'''<span class="mapping-item">
                <span class="mapping-original">{orig}</span>
                <span class="mapping-arrow">→</span>
                <span class="mapping-target">{target}</span>
                <span class="mapping-confidence">({conf_text})</span>
            </span>'''
        st.markdown(f'<div style="margin-bottom:1rem">{mapping_html}</div>', unsafe_allow_html=True)
    
    # Check required columns
    required = ["voltage", "current", "temperature", "soc", "cycle"]
    missing = [c for c in required if c not in df_raw.columns]
    
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
        
        # Manual mapping UI
        st.markdown("#### 🔧 Manual Column Assignment")
        available = [c for c in df_raw.columns if c not in col_map.values()]
        
        manual_map = {}
        cols = st.columns(len(missing))
        for i, m in enumerate(missing):
            with cols[i]:
                choice = st.selectbox(f"{m}", ["(skip)"] + available, key=f"map_{m}")
                if choice != "(skip)":
                    manual_map[m] = choice
        
        if st.button("Apply Mapping", type="primary"):
            for target, orig in manual_map.items():
                df_raw = df_raw.rename(columns={orig: target})
            st.rerun()
        st.stop()
    
    # Preview
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df_raw.head(10), use_container_width=True, height=200)
    
    # Process Data
    df_fe = feature_engineering(df_raw, window)
    df_out = build_models_and_scores(df_fe, contamination)
    
    # Calculate metrics
    avg_score = df_out["battery_health_score"].mean()
    high_risk_pct = (df_out["battery_health_score"] < 60).mean() * 100
    anomaly_pct = df_out["anomaly_flag"].mean() * 100
    status, color, icon = get_status(avg_score)
    
    # Dashboard Layout
    st.markdown("### 📊 Health Overview")
    
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f'''<div class="metric-box">
                <div class="metric-value" style="color:{color}">{avg_score:.1f}</div>
                <div class="metric-label">Health Score</div>
            </div>''', unsafe_allow_html=True)
        
        with m2:
            st.markdown(f'''<div class="metric-box">
                <div class="metric-value">{high_risk_pct:.1f}%</div>
                <div class="metric-label">High Risk Time</div>
            </div>''', unsafe_allow_html=True)
        
        with m3:
            st.markdown(f'''<div class="metric-box">
                <div class="metric-value">{anomaly_pct:.1f}%</div>
                <div class="metric-label">Anomalies</div>
            </div>''', unsafe_allow_html=True)
        
        with m4:
            st.markdown(f'''<div class="metric-box">
                <div class="metric-value">{len(df_out):,}</div>
                <div class="metric-label">Data Points</div>
            </div>''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        status_class = f"status-{status.lower()}"
        st.markdown(f'<div class="status-badge {status_class}">{icon} {status}</div>', 
                   unsafe_allow_html=True)
        
        # Compact gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(30,41,59,0.5)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,68,68,0.2)"},
                    {"range": [50, 80], "color": "rgba(245,158,11,0.2)"},
                    {"range": [80, 100], "color": "rgba(16,185,129,0.2)"},
                ],
            },
            number={"suffix": "%", "font": {"size": 28, "color": "#f1f5f9"}},
        ))
        fig.update_layout(
            height=180, margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#94a3b8"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Timeline", "🚨 Anomalies", "📄 Export"])
    
    with tab1:
        fig_health = px.area(df_out, x="time", y="battery_health_score",
                             labels={"time": "Time", "battery_health_score": "Health Score"})
        fig_health.update_traces(fill='tozeroy', line_color='#3b82f6', 
                                  fillcolor='rgba(59,130,246,0.2)')
        fig_health.update_layout(
            height=350, margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)", range=[0, 100]),
            font={"color": "#94a3b8"}
        )
        st.plotly_chart(fig_health, use_container_width=True)
        
        st.markdown('''<div class="tip-box">
            💡 <strong>Tip:</strong> Sudden drops indicate thermal stress or voltage sag. 
            Monitor patterns during charging cycles and heavy loads.
        </div>''', unsafe_allow_html=True)
    
    with tab2:
        anomalies = df_out[df_out["anomaly_flag"] == 1]
        
        col_a, col_b = st.columns([3, 1])
        
        with col_a:
            fig_volt = px.line(df_out, x="time", y="voltage", 
                              labels={"time": "Time", "voltage": "Voltage (V)"})
            fig_volt.update_traces(line_color='#10b981')
            
            if len(anomalies) > 0:
                fig_volt.add_scatter(x=anomalies["time"], y=anomalies["voltage"],
                                     mode="markers", name="Anomaly",
                                     marker=dict(color="#ef4444", size=10, symbol="x"))
            
            fig_volt.update_layout(
                height=300, margin=dict(l=40, r=20, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                font={"color": "#94a3b8"}, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_volt, use_container_width=True)
        
        with col_b:
            st.markdown(f'''<div class="card">
                <div class="metric-value" style="color:#ef4444">{len(anomalies)}</div>
                <div class="metric-label">Anomalies Found</div>
                <hr style="border-color:rgba(148,163,184,0.2);margin:0.8rem 0">
                <div style="font-size:0.8rem;color:#94a3b8">
                    Detection uses Isolation Forest algorithm on voltage, 
                    current, temperature patterns.
                </div>
            </div>''', unsafe_allow_html=True)
        
        # Critical moments table
        st.markdown("#### ⚠️ Critical Moments")
        worst = df_out.nsmallest(10, "battery_health_score")[
            ["time", "voltage", "temperature", "soc", "battery_health_score"]
        ].round(2)
        worst.columns = ["Time", "Voltage", "Temp", "SoC", "Health"]
        st.dataframe(worst, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("#### 📥 Download Processed Data")
        
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "cellguard_output.csv", "text/csv", 
                          use_container_width=True)
        
        st.markdown('''<div class="tip-box">
            📊 <strong>Output includes:</strong> Original data + health scores, anomaly flags, 
            risk predictions, and rolling features for further ML analysis.
        </div>''', unsafe_allow_html=True)
        
        with st.expander("Preview processed data"):
            st.dataframe(df_out.head(50), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("🔋 CellGuard.AI — Predicting battery failures before they happen")


if __name__ == "__main__":
    main()
