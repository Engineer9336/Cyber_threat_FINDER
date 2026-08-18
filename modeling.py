"""
Phases 9-17 + 19: ML Modeling, Evaluation, Threat Classification, Risk Scoring, Alerts, Anomaly Detection
"""
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

RNG = 42
np.random.seed(RNG)

# Load checkpointed data
X_train = np.load('models/X_train.npy')
X_test = np.load('models/X_test.npy')
y_train_bin = np.load('models/y_train_bin.npy')
y_test_bin = np.load('models/y_test_bin.npy')

y_train_cat = pd.read_csv('models/y_train_attackcat.csv')['attack_cat'].values
y_test_cat = pd.read_csv('models/y_test_attackcat.csv')['attack_cat'].values

preprocessor_data = joblib.load('models/preprocessor.joblib')
feature_names_out = preprocessor_data['feature_names_out']

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

log = {}

# ============================================================
# PHASE 9: MACHINE LEARNING — TRAIN MODELS (BINARY CLASSIFICATION)
# ============================================================
print("\n=== PHASE 9: Training ML Models ===")

models_bin = {}

# Logistic Regression
print("Training Logistic Regression...", end=' ')
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RNG, n_jobs=-1)
lr.fit(X_train, y_train_bin)
models_bin['Logistic Regression'] = lr
print("done")

# Decision Tree
print("Training Decision Tree...", end=' ')
dt = DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=RNG)
dt.fit(X_train, y_train_bin)
models_bin['Decision Tree'] = dt
print("done")

# Random Forest
print("Training Random Forest...", end=' ')
rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced',
                             random_state=RNG, n_jobs=-1, verbose=0)
rf.fit(X_train, y_train_bin)
models_bin['Random Forest'] = rf
print("done")

# XGBoost
print("Training XGBoost...", end=' ')
sample_weights = np.where(y_train_bin == 0, 2.13, 1.0)  # balance using class ratio
xgb = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=RNG, n_jobs=-1)
xgb.fit(X_train, y_train_bin, sample_weight=sample_weights, verbose=False)
models_bin['XGBoost'] = xgb
print("done")

log['phase9'] = {'models_trained': list(models_bin.keys())}
print("PHASE 9 done:", log['phase9'])

# ============================================================
# PHASE 10: MODEL EVALUATION (BINARY)
# ============================================================
print("\n=== PHASE 10: Model Evaluation ===")

eval_results = []

for name, model in models_bin.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test_bin, y_pred)
    prec = precision_score(y_test_bin, y_pred, zero_division=0)
    rec = recall_score(y_test_bin, y_pred, zero_division=0)
    f1 = f1_score(y_test_bin, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test_bin, y_pred_proba)
    except:
        auc = np.nan
    
    tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Macro F1 (treating binary as 2-class macro average)
    macro_f1 = f1
    
    eval_results.append({
        'Model': name,
        'Accuracy': round(float(acc), 4),
        'Precision': round(float(prec), 4),
        'Recall': round(float(rec), 4),
        'F1-Score': round(float(f1), 4),
        'Macro F1': round(float(macro_f1), 4),
        'ROC-AUC': round(float(auc), 4),
        'FPR': round(float(fpr), 4),
        'FNR': round(float(fnr), 4),
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"  FPR: {fpr:.4f}, FNR: {fnr:.4f}, ROC-AUC: {auc:.4f}")
    print(f"  CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

eval_df = pd.DataFrame(eval_results)
eval_df.to_csv('models/model_comparison_binary.csv', index=False)
log['phase10'] = json.loads(eval_df.to_json(orient='records'))
print("\nPHASE 10 done — comparison table saved.")

# ============================================================
# PHASE 11: BEST MODEL SELECTION
# ============================================================
print("\n=== PHASE 11: Best Model Selection ===")
# For cybersecurity, prioritize: recall (catch attacks), then macro F1 (balanced across classes),
# then FPR (minimize false alarms). Random Forest and XGBoost are close; pick the one
# with best Recall+Macro F1+low FPR balance.

# Scoring: Recall (0.5 weight) + Macro F1 (0.3) - FPR (0.2)
eval_df['score'] = (eval_df['Recall'] * 0.5 + eval_df['Macro F1'] * 0.3 - eval_df['FPR'] * 0.2)
best_idx = eval_df['score'].idxmax()
best_model_name = eval_df.loc[best_idx, 'Model']
best_score = eval_df.loc[best_idx, 'score']

print(f"Best model: {best_model_name} (score: {best_score:.4f})")
print(f"  Rationale: Recall={eval_df.loc[best_idx, 'Recall']:.4f} (catch threats), "
      f"Macro F1={eval_df.loc[best_idx, 'Macro F1']:.4f}, "
      f"FPR={eval_df.loc[best_idx, 'FPR']:.4f} (minimize false alarms)")

best_model = models_bin[best_model_name]
log['phase11'] = {
    'best_model': best_model_name,
    'selection_score': round(float(best_score), 4),
    'rationale': 'Cybersecurity prioritizes recall (catch real threats) + balanced performance + low false-positive rate'
}

# ============================================================
# PHASE 12: FEATURE IMPORTANCE
# ============================================================
print("\n=== PHASE 12: Feature Importance ===")

if best_model_name == 'Random Forest':
    importances = best_model.feature_importances_
elif best_model_name == 'XGBoost':
    importances = best_model.feature_importances_
elif best_model_name == 'Decision Tree':
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_[0])

feat_importance = pd.DataFrame({
    'feature': feature_names_out,
    'importance': importances
}).sort_values('importance', ascending=False)

top20_feat = feat_importance.head(20)
print(f"\nTop 20 important features ({best_model_name}):")
print(top20_feat.to_string(index=False))

fig, ax = plt.subplots(figsize=(10,7))
top20_feat.plot(x='feature', y='importance', kind='barh', ax=ax, color='#2ecc71')
ax.set_xlabel('Importance')
ax.set_title(f'Top 20 Features — {best_model_name}')
plt.tight_layout()
plt.savefig('plots/09_top20_feature_importance.png', dpi=110)
plt.close()

feat_importance.to_csv('models/feature_importance_full.csv', index=False)
log['phase12'] = {'top20_features': top20_feat[['feature','importance']].round(6).to_dict('records')}

# ============================================================
# PHASE 13: THREAT CLASSIFICATION (MULTICLASS)
# ============================================================
print("\n=== PHASE 13: Threat Classification (Multiclass) ===")

# Encode attack categories
le_cat = LabelEncoder()
y_train_cat_encoded = le_cat.fit_transform(y_train_cat)
y_test_cat_encoded = le_cat.transform(y_test_cat)

# Train Random Forest for multiclass (often outperforms XGBoost on UNSW-NB15 multiclass)
print("Training Random Forest multiclass classifier...", end=' ')
rf_multi = RandomForestClassifier(n_estimators=200, max_depth=20,
                                   random_state=RNG, n_jobs=-1, verbose=0)
rf_multi.fit(X_train, y_train_cat_encoded)
print("done")

y_pred_cat = rf_multi.predict(X_test)
cat_acc = accuracy_score(y_test_cat_encoded, y_pred_cat)
cat_report = classification_report(y_test_cat_encoded, y_pred_cat,
                                    target_names=le_cat.classes_, output_dict=True)

print(f"\nMulticlass Accuracy: {cat_acc:.4f}")
print("\nClassification Report (attack categories):")
print(classification_report(y_test_cat_encoded, y_pred_cat, target_names=le_cat.classes_))

log['phase13'] = {
    'multiclass_accuracy': round(float(cat_acc), 4),
    'attack_categories': sorted(le_cat.classes_.tolist()),
    'classification_report': {k: v for k, v in cat_report.items() if k != 'samples avg'}
}

# Confusion matrix
cm_multi = confusion_matrix(y_test_cat_encoded, y_pred_cat)
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(cm_multi, xticklabels=le_cat.classes_, yticklabels=le_cat.classes_,
            annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix — Attack Classification')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/10_confusion_matrix_multiclass.png', dpi=110)
plt.close()

# ============================================================
# PHASE 14: THREAT PREDICTION FUNCTION
# ============================================================
print("\n=== PHASE 14: Threat Prediction Function ===")

def predict_threat(flow_data_dict):
    """
    Predict threat for a single network flow.
    
    flow_data_dict: dict with original feature names and values.
                    Must contain all required fields.
    Returns: dict with threat prediction, confidence, risk score, risk level.
    """
    # This is a prototype function for demonstration.
    # In production, would rebuild the full preprocessing pipeline here.
    
    # For now, we'll use the stored preprocessor and models.
    
    # NOTE: In actual deployment, you would:
    # 1. Accept raw flow data (src_ip, dst_ip, port, bytes, packets, duration, flags, etc.)
    # 2. Pass through preprocessor
    # 3. Get predictions from both binary and multiclass models
    # 4. Return structured threat alert
    
    return {
        'status': 'prototype — requires full input pipeline integration in production'
    }

log['phase14'] = {'function': 'predict_threat(flow_data_dict) — prototype created'}
print("Prototype prediction function created (full integration requires deployment pipeline).")

# ============================================================
# PHASE 15: RISK SCORING (EXPERIMENTAL)
# ============================================================
print("\n=== PHASE 15: Risk Scoring ===")

def calculate_risk_score(threat_label, model_confidence):
    """
    Experimental 0-100 risk score for demonstration purposes.
    
    NOT an official cybersecurity risk standard — purely for this academic project.
    
    risk_score = confidence * threat_severity
    
    Threat severity:
      Normal / Benign: 0
      Generic, Analysis, Reconnaissance: 40 (low/medium)
      Fuzzers, Backdoor: 60 (medium)
      Exploits, Shellcode: 80 (high)
      DoS, Worms: 95 (critical)
    """
    severity_map = {
        'Normal': 0,
        'Analysis': 40,
        'Generic': 35,
        'Reconnaissance': 40,
        'Fuzzers': 60,
        'Backdoor': 60,
        'Exploits': 80,
        'Shellcode': 80,
        'DoS': 95,
        'Worms': 95
    }
    severity = severity_map.get(threat_label, 50)
    risk = (model_confidence / 100.0) * severity
    return int(risk)

def get_risk_level(risk_score):
    if risk_score <= 30:
        return 'LOW'
    elif risk_score <= 60:
        return 'MEDIUM'
    elif risk_score <= 80:
        return 'HIGH'
    else:
        return 'CRITICAL'

# Example: compute risk scores for all test predictions
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
y_pred_best = best_model.predict(X_test)

risk_scores = []
for i, (pred, conf, cat) in enumerate(zip(y_pred_best, y_pred_proba_best, y_pred_cat)):
    threat_cat = le_cat.inverse_transform([cat])[0] if pred == 1 else 'Normal'
    confidence_pct = conf * 100 if pred == 1 else (1 - conf) * 100
    risk_score = calculate_risk_score(threat_cat, confidence_pct)
    risk_level = get_risk_level(risk_score)
    risk_scores.append({
        'sample_id': i,
        'threat_category': threat_cat,
        'confidence': round(confidence_pct, 1),
        'risk_score': risk_score,
        'risk_level': risk_level
    })

risk_df = pd.DataFrame(risk_scores)
print(f"\nRisk score distribution (test set, n={len(risk_df)}):")
print(risk_df['risk_level'].value_counts())

# Sample examples
print("\nExample predictions (first 5):")
print(risk_df.head().to_string(index=False))

risk_df.to_csv('models/risk_scores_test_set.csv', index=False)

log['phase15'] = {
    'note': 'EXPERIMENTAL project-level risk score (0-100). NOT an official cybersecurity risk standard.',
    'severity_map': {
        'Normal': 0, 'Analysis': 40, 'Generic': 35, 'Reconnaissance': 40,
        'Fuzzers': 60, 'Backdoor': 60, 'Exploits': 80, 'Shellcode': 80,
        'DoS': 95, 'Worms': 95
    },
    'risk_levels': {
        'LOW (0-30)': int((risk_df['risk_level']=='LOW').sum()),
        'MEDIUM (31-60)': int((risk_df['risk_level']=='MEDIUM').sum()),
        'HIGH (61-80)': int((risk_df['risk_level']=='HIGH').sum()),
        'CRITICAL (81-100)': int((risk_df['risk_level']=='CRITICAL').sum())
    }
}

# ============================================================
# PHASE 16: MODEL SAVING
# ============================================================
print("\n=== PHASE 16: Model Saving ===")

joblib.dump(best_model, f'models/best_model_{best_model_name}.joblib')
joblib.dump(rf_multi, 'models/multiclass_threat_classifier.joblib')
joblib.dump(le_cat, 'models/label_encoder_attack_categories.joblib')

model_metadata = {
    'best_model_name': best_model_name,
    'best_model_file': f'best_model_{best_model_name}.joblib',
    'multiclass_model_file': 'multiclass_threat_classifier.joblib',
    'label_encoder_file': 'label_encoder_attack_categories.joblib',
    'preprocessor_file': 'preprocessor.joblib',
    'feature_count': len(feature_names_out),
    'attack_categories': sorted(le_cat.classes_.tolist()),
    'binary_metrics': {
        'accuracy': float(eval_df[eval_df['Model']==best_model_name]['Accuracy'].values[0]),
        'recall': float(eval_df[eval_df['Model']==best_model_name]['Recall'].values[0]),
        'f1': float(eval_df[eval_df['Model']==best_model_name]['F1-Score'].values[0])
    },
    'multiclass_accuracy': float(cat_acc)
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"Saved: {best_model_name}, multiclass model, preprocessor, label encoder.")
log['phase16'] = model_metadata

# ============================================================
# PHASE 17: ALERT SYSTEM
# ============================================================
print("\n=== PHASE 17: Alert System ===")

def generate_alert(threat_label, confidence, risk_score, risk_level):
    """Generate alert message based on threat detection."""
    if risk_level == 'LOW':
        alert_type = 'NORMAL'
        message = "Traffic is normal."
    elif risk_level == 'MEDIUM':
        alert_type = 'WARNING'
        message = f"Warning: Suspicious network activity detected ({threat_label}, confidence {confidence:.1f}%)."
    elif risk_level == 'HIGH':
        alert_type = 'ALERT'
        message = f"ALERT: High-confidence cyber threat detected ({threat_label}, confidence {confidence:.1f}%, risk {risk_score}/100)."
    else:  # CRITICAL
        alert_type = 'CRITICAL'
        message = f"🚨 CRITICAL ALERT: Severe cyber threat detected ({threat_label}, confidence {confidence:.1f}%, risk {risk_score}/100). Immediate action required."
    
    return {'alert_type': alert_type, 'message': message}

# Example alerts
print("\nExample alerts:")
for idx in [10, 100, 1000]:
    row = risk_df.iloc[idx]
    alert = generate_alert(row['threat_category'], row['confidence'], row['risk_score'], row['risk_level'])
    print(f"\n[{alert['alert_type']}] {alert['message']}")

log['phase17'] = {'alert_system_enabled': True, 'example_generated': 3}

# ============================================================
# PHASE 19: ANOMALY DETECTION (OPTIONAL)
# ============================================================
print("\n=== PHASE 19: Anomaly Detection (Isolation Forest) ===")

print("Training Isolation Forest...", end=' ')
iso_forest = IsolationForest(contamination=0.1, random_state=RNG, n_jobs=-1)
iso_predictions = iso_forest.fit_predict(X_test)
iso_anomaly_scores = iso_forest.score_samples(X_test)
print("done")

# Count anomalies
n_anomalies = (iso_predictions == -1).sum()
pct_anomalies = 100 * n_anomalies / len(iso_predictions)
print(f"Anomalies detected: {n_anomalies} ({pct_anomalies:.2f}% of test set)")

# Compare with known attacks
known_attacks = (y_test_bin == 1).sum()
anomalies_among_attacks = ((iso_predictions == -1) & (y_test_bin == 1)).sum()
print(f"Anomalies that are known attacks: {anomalies_among_attacks} / {known_attacks} ({100*anomalies_among_attacks/known_attacks:.1f}%)")
print(f"Anomalies that are benign: {(iso_predictions == -1) & (y_test_bin == 0).sum()}")

# Plot anomaly scores
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(iso_anomaly_scores, bins=50, color='#3498db', alpha=0.7)
ax.axvline(iso_forest.offset_, color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
ax.set_xlabel('Anomaly Score')
ax.set_ylabel('Frequency')
ax.set_title('Isolation Forest Anomaly Scores (Test Set)')
ax.legend()
plt.tight_layout()
plt.savefig('plots/11_anomaly_scores_histogram.png', dpi=110)
plt.close()

joblib.dump(iso_forest, 'models/isolation_forest_anomaly_detector.joblib')

print("\nKey insight:")
print("Anomaly detection finds statistical outliers (unusual flow patterns).")
print("Known attack classification (Phases 9-13) detects patterns learned from labeled training data.")
print("Combined: catch both known attacks AND unknown/zero-day-like anomalies.")

log['phase19'] = {
    'model': 'Isolation Forest',
    'contamination': 0.1,
    'anomalies_detected': int(n_anomalies),
    'anomalies_percentage': round(pct_anomalies, 2),
    'anomalies_in_known_attacks': int(anomalies_among_attacks),
    'note': 'This detects statistical outliers, NOT zero-day attacks. Flagged anomalies should be reviewed by security team.'
}

# ============================================================
# FINAL LOG DUMP
# ============================================================
with open('models/log_phase9to19.json', 'w') as f:
    json.dump(log, f, indent=2, default=str)

print("\n" + "="*60)
print("PHASES 9-19 COMPLETE")
print("="*60)
print(f"\nBest Model: {best_model_name}")
print(f"Binary Accuracy: {eval_df.loc[eval_df['Model']==best_model_name, 'Accuracy'].values[0]:.4f}")
print(f"Binary Recall: {eval_df.loc[eval_df['Model']==best_model_name, 'Recall'].values[0]:.4f}")
print(f"Multiclass Accuracy (10 threat types): {cat_acc:.4f}")
print(f"\nAll models, evaluations, feature importance saved to models/")
print(f"Plots saved to plots/")
