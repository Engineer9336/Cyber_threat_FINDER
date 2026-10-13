"""
Cyber Threat Finder — Threat Dashboard
Save this as: pages/1_Threat_Dashboard.py

Expected project-root artifacts:
model_metadata.json, best_model_XGBoost.joblib, preprocessor.joblib,
label_encoder_attack_categories.joblib, isolation_forest_anomaly_detector.joblib,
risk_scores_test_set.csv, feature_importance_full.csv
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent

st.title("🛡️ Cyber Threat Finder")
st.caption("AI-powered cyber threat analytics and risk overview")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {
  background: rgba(80, 100, 130, 0.10);
  border: 1px solid rgba(128, 128, 128, 0.25);
  padding: 14px 16px; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return {}


def read_csv(path):
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def find_col(df, names):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    for col in df.columns:
        if any(n.lower() in str(col).lower() for n in names):
            return col
    return None


def find_metric(obj, names):
    if not isinstance(obj, dict):
        return None
    for key, val in obj.items():
        if str(key).lower() in [n.lower() for n in names] and isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            found = find_metric(val, names)
            if found is not None:
                return found
    return None


metadata = read_json(ROOT / "model_metadata.json")
risk_df = read_csv(ROOT / "risk_scores_test_set.csv")
importance_df = read_csv(ROOT / "feature_importance_full.csv")

# Check/load the saved model artifacts without making the analytics page depend on them.
artifact_files = [
    "best_model_XGBoost.joblib",
    "preprocessor.joblib",
    "label_encoder_attack_categories.joblib",
    "isolation_forest_anomaly_detector.joblib",
]
for filename in artifact_files:
    path = ROOT / filename
    if path.exists():
        try:
            joblib.load(path)
        except Exception as exc:
            st.warning(f"Could not load {filename}: {exc}")

risk_col = find_col(risk_df, ["risk_score", "risk score", "risk", "predicted_risk", "risk_level"])
attack_col = find_col(risk_df, ["attack_category", "attack category", "attack_type", "attack type", "label", "class"])
anomaly_col = find_col(risk_df, ["anomaly", "is_anomaly", "anomaly_flag", "anomaly_score"])

high_risk = 0
if risk_col and not risk_df.empty:
    if pd.api.types.is_numeric_dtype(risk_df[risk_col]):
        high_risk = int((pd.to_numeric(risk_df[risk_col], errors="coerce") >= 70).sum())
    else:
        high_risk = int(risk_df[risk_col].astype(str).str.lower().isin(["high", "critical", "severe"]).sum())

category_count = int(risk_df[attack_col].nunique()) if attack_col and not risk_df.empty else 0
anomalies = 0
if anomaly_col and not risk_df.empty:
    values = risk_df[anomaly_col]
    if pd.api.types.is_numeric_dtype(values):
        anomalies = int((pd.to_numeric(values, errors="coerce") == -1).sum())
    else:
        anomalies = int(values.astype(str).str.lower().isin(["true", "1", "yes", "anomaly", "anomalous"]).sum())

a, b, c, d = st.columns(4)
a.metric("Test records", f"{len(risk_df):,}")
b.metric("High-risk records", f"{high_risk:,}")
c.metric("Attack categories", f"{category_count:,}")
d.metric("Potential anomalies", f"{anomalies:,}")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("📊 Attack Category Distribution")
    if attack_col and not risk_df.empty:
        counts = risk_df[attack_col].fillna("Unknown").astype(str).value_counts().rename_axis("Attack category").reset_index(name="Records")
        fig = px.bar(counts, x="Attack category", y="Records", color="Attack category", title="Records by category")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Records")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Category chart unavailable: no recognizable category column in risk_scores_test_set.csv.")

with right:
    st.subheader("⚠️ Risk Score Distribution")
    if risk_col and not risk_df.empty:
        if pd.api.types.is_numeric_dtype(risk_df[risk_col]):
            values = pd.to_numeric(risk_df[risk_col], errors="coerce").dropna()
            if not values.empty:
                fig = px.histogram(x=values, nbins=20, title="Risk score frequency")
                fig.update_layout(xaxis_title="Risk score", yaxis_title="Records")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Risk column has no numeric values.")
        else:
            counts = risk_df[risk_col].fillna("Unknown").astype(str).value_counts().rename_axis("Risk level").reset_index(name="Records")
            fig = px.pie(counts, names="Risk level", values="Records", hole=0.35, title="Risk-level split")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Risk chart unavailable: no recognizable risk column in risk_scores_test_set.csv.")

st.subheader("🧠 Model Performance")
metrics = {
    "Accuracy": ["accuracy", "test_accuracy"],
    "Precision": ["precision", "test_precision"],
    "Recall": ["recall", "test_recall"],
    "F1 score": ["f1", "f1_score", "f1-score"],
    "ROC-AUC": ["roc_auc", "roc-auc", "auc"],
}
available = [(label, find_metric(metadata, keys)) for label, keys in metrics.items()]
available = [(label, val) for label, val in available if val is not None]
if available:
    cols = st.columns(min(5, len(available)))
    for col, (label, val) in zip(cols, available):
        col.metric(label, f"{val:.3f}")
else:
    st.caption("Metrics were not found in model_metadata.json.")

st.subheader("🔎 Feature Importance")
if not importance_df.empty:
    feature_col = find_col(importance_df, ["feature", "feature_name", "name"])
    importance_col = find_col(importance_df, ["importance", "importance_score", "score", "weight"])
    if feature_col and importance_col:
        plot_df = importance_df[[feature_col, importance_col]].copy()
        plot_df[importance_col] = pd.to_numeric(plot_df[importance_col], errors="coerce")
        plot_df = plot_df.dropna().sort_values(importance_col, ascending=False).head(15)
        if not plot_df.empty:
            fig = px.bar(plot_df.sort_values(importance_col), x=importance_col, y=feature_col, orientation="h", title="Top 15 features")
            fig.update_layout(xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(importance_df, use_container_width=True)
    else:
        st.dataframe(importance_df.head(20), use_container_width=True)
else:
    st.info("feature_importance_full.csv not found or unavailable.")

with st.expander("📄 View risk-score records"):
    if not risk_df.empty:
        st.dataframe(risk_df, use_container_width=True)
        st.download_button(
            "Download risk records CSV",
            risk_df.to_csv(index=False).encode("utf-8"),
            file_name="risk_scores_test_set.csv",
            mime="text/csv",
        )
    else:
        st.info("risk_scores_test_set.csv not found in the project root.")

with st.expander("🧰 Artifact status"):
    expected = [
        "model_metadata.json", "best_model_XGBoost.joblib", "preprocessor.joblib",
        "label_encoder_attack_categories.joblib", "isolation_forest_anomaly_detector.joblib",
        "risk_scores_test_set.csv", "feature_importance_full.csv",
    ]
    st.dataframe(pd.DataFrame([
        {"Artifact": name, "Status": "Found" if (ROOT / name).exists() else "Missing"}
        for name in expected
    ]), hide_index=True, use_container_width=True)

st.caption("Educational project dashboard. Results depend on saved model/test artifacts and are not a substitute for production security monitoring.")

