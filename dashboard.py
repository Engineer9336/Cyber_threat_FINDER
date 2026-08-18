"""
PHASE 18: Streamlit Dashboard
AI-Based Cyber Threat Detection Framework
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Page config
st.set_page_config(page_title="Cyber Threat Detection Dashboard", layout="wide")

# Load models and metadata
@st.cache_resource
def load_models():
    model_metadata = json.load(open('models/model_metadata.json'))
    best_model = joblib.load(f"models/{model_metadata['best_model_file']}")
    preprocessor = joblib.load('models/preprocessor.joblib')
    le_cat = joblib.load('models/label_encoder_attack_categories.joblib')
    iso_forest = joblib.load('models/isolation_forest_anomaly_detector.joblib')
    risk_scores_df = pd.read_csv('models/risk_scores_test_set.csv')
    return model_metadata, best_model, preprocessor, le_cat, iso_forest, risk_scores_df

model_metadata, best_model, preprocessor, le_cat, iso_forest, risk_scores_df = load_models()

# ============================================================
# Dashboard Header
# ============================================================
st.markdown("# 🛡️ AI-Based Cyber Threat Detection Dashboard")
st.markdown("Academic Project | UNSW-NB15 Dataset | Real ML Models")
st.markdown("---")

# ============================================================
# KPI Metrics
# ============================================================
col1, col2, col3, col4 = st.columns(4)

total_analyzed = len(risk_scores_df)
benign_count = (risk_scores_df['risk_level'] == 'LOW').sum()
threat_count = (risk_scores_df['risk_level'].isin(['MEDIUM', 'HIGH', 'CRITICAL'])).sum()
critical_count = (risk_scores_df['risk_level'] == 'CRITICAL').sum()

with col1:
    st.metric("Total Traffic Analyzed", f"{total_analyzed:,}")
with col2:
    st.metric("Benign Traffic", f"{benign_count:,}", f"{100*benign_count/total_analyzed:.1f}%")
with col3:
    st.metric("Threats Detected", f"{threat_count:,}", f"{100*threat_count/total_analyzed:.1f}%")
with col4:
    st.metric("Critical Alerts", f"{critical_count}", f"{100*critical_count/total_analyzed:.2f}%")

st.markdown("---")

# ============================================================
# Threat Distribution
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Threat Category Distribution")
    threat_dist = risk_scores_df[risk_scores_df['risk_level'] != 'LOW']['threat_category'].value_counts().head(10)
    fig, ax = plt.subplots()
    threat_dist.plot(kind='barh', ax=ax, color='#e74c3c')
    ax.set_xlabel('Count')
    st.pyplot(fig)

with col2:
    st.subheader("⚠️ Risk Level Distribution")
    risk_dist = risk_scores_df['risk_level'].value_counts()
    colors = {'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e67e22', 'CRITICAL': '#c0392b'}
    fig, ax = plt.subplots()
    risk_dist.plot(kind='bar', ax=ax, color=[colors.get(x, '#95a5a6') for x in risk_dist.index])
    ax.set_ylabel('Count')
    st.pyplot(fig)

st.markdown("---")

# ============================================================
# Model Performance Metrics
# ============================================================
st.subheader("🎯 Model Performance")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", model_metadata['best_model_name'])
with col2:
    st.metric("Binary Accuracy", f"{model_metadata['binary_metrics']['accuracy']:.2%}")
with col3:
    st.metric("Recall", f"{model_metadata['binary_metrics']['recall']:.2%}")
with col4:
    st.metric("F1-Score", f"{model_metadata['binary_metrics']['f1']:.2%}")

col1, col2 = st.columns(2)
with col1:
    st.metric("Multiclass Accuracy (10 threats)", f"{model_metadata['multiclass_accuracy']:.2%}")
with col2:
    st.metric("Attack Categories", len(model_metadata['attack_categories']))

st.markdown("---")

# ============================================================
# Recent Alerts
# ============================================================
st.subheader("🚨 Sample Threat Alerts (Latest Critical/High)")

alerts_df = risk_scores_df[risk_scores_df['risk_level'].isin(['CRITICAL', 'HIGH'])].tail(10).copy()
alerts_df['Alert'] = alerts_df.apply(
    lambda row: f"[{row['risk_level']}] {row['threat_category']} - Confidence: {row['confidence']:.1f}% | Risk: {row['risk_score']}/100",
    axis=1
)

for idx, row in alerts_df.iterrows():
    color = '#c0392b' if row['risk_level'] == 'CRITICAL' else '#e67e22'
    st.markdown(f"<span style='color:{color}'>{row['Alert']}</span>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# Feature Importance
# ============================================================
st.subheader("⭐ Top 15 Important Features")

feat_importance_df = pd.read_csv('models/feature_importance_full.csv').head(15)
fig, ax = plt.subplots(figsize=(10, 5))
feat_importance_df.sort_values('importance', ascending=True).plot(
    x='feature', y='importance', kind='barh', ax=ax, legend=False, color='#3498db'
)
ax.set_xlabel('Importance Score')
st.pyplot(fig)

st.markdown("---")

# ============================================================
# Risk Score Breakdown
# ============================================================
st.subheader("📈 Risk Score Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("**Risk Distribution Statistics**")
    risk_stats = risk_scores_df['risk_score'].describe()
    st.write(risk_stats.round(2))

with col2:
    st.write("**Risk Level Breakdown**")
    risk_breakdown = risk_scores_df['risk_level'].value_counts()
    for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        count = risk_breakdown.get(level, 0)
        pct = 100 * count / len(risk_scores_df)
        st.write(f"{level}: {count:,} ({pct:.1f}%)")

st.markdown("---")

# ============================================================
# System Info
# ============================================================
st.subheader("ℹ️ System Information")

col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**Features**: {model_metadata['feature_count']}")
    st.write(f"**Dataset**: UNSW-NB15")
with col2:
    st.write(f"**Training Samples**: 175,341")
    st.write(f"**Test Samples**: 82,332")
with col3:
    st.write(f"**Attack Categories**: {len(model_metadata['attack_categories'])}")
    st.write(f"**Binary Classification**: Benign vs. Malicious")

st.markdown("---")

# ============================================================
# Disclaimer
# ============================================================
st.info("""
### ⚠️ Disclaimer
This dashboard is an **academic project prototype** for cybersecurity research and AI/ML demonstration.

- **Risk scores** are experimental (0-100 scale) and NOT official cybersecurity risk standards.
- **Anomaly detection** identifies statistical outliers, not zero-day attacks.
- **For production use**, this system requires:
  - Real-time network packet capture integration
  - SOC alerting pipeline
  - Security team review and response workflows
  - Additional hardening and validation

**This system is for DEFENSIVE cybersecurity research only.**
""")

st.markdown("---")
st.markdown("Built with ❤️ for academic research | Python | scikit-learn | XGBoost | Streamlit")
