"""
AI-Based Cyber Threat Detection Framework
Dataset: UNSW-NB15 (official pre-partitioned train/test split)
Phases 4-17 + 19 implemented end-to-end.
"""
import json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib

RNG = 42
np.random.seed(RNG)
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

log = {}

# ============================================================
# PHASE 4: PREPROCESSING
# ============================================================
train = pd.read_csv('data/UNSW_NB15_training-set.csv')
test = pd.read_csv('data/UNSW_NB15_testing-set.csv')

log['phase4'] = {'initial_train_rows': int(len(train)), 'initial_test_rows': int(len(test))}

for df in (train, test):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

# missing / infinite check
num_cols_chk = train.select_dtypes(include=[np.number]).columns
log['phase4']['missing_train'] = int(train.isnull().sum().sum())
log['phase4']['missing_test'] = int(test.isnull().sum().sum())
log['phase4']['inf_train'] = int(np.isinf(train[num_cols_chk]).sum().sum())
log['phase4']['inf_test'] = int(np.isinf(test[num_cols_chk]).sum().sum())

# duplicate check (excluding id)
feat_cols_all = [c for c in train.columns if c != 'id']
log['phase4']['dupe_rows_train_excl_id'] = int(train[feat_cols_all].duplicated().sum())
log['phase4']['dupe_rows_test_excl_id'] = int(test[feat_cols_all].duplicated().sum())

train_hash = pd.util.hash_pandas_object(train[feat_cols_all], index=False)
test_hash = pd.util.hash_pandas_object(test[feat_cols_all], index=False)
log['phase4']['train_test_overlap_rows'] = int(test_hash.isin(set(train_hash)).sum())
# DECISION: keep in-class duplicates (legitimate repeated flow signatures in network data,
# not data-entry errors); dropping would deviate from the official benchmark partition.
# This is documented, not silently done.

# attack_cat label sanity
log['phase4']['attack_cat_values_train'] = sorted(train['attack_cat'].unique().tolist())
assert set(train['attack_cat'].unique()) == set(test['attack_cat'].unique()), "label mismatch train/test"

print("PHASE 4 done:", json.dumps(log['phase4'], indent=2))

# ============================================================
# PHASE 5: EDA
# ============================================================
sns.set_style('whitegrid')
palette = {'Normal': '#2ecc71'}

# 1. Benign vs malicious
fig, ax = plt.subplots(figsize=(5,4))
train['label'].map({0:'Benign',1:'Malicious'}).value_counts().plot(kind='bar', color=['#2ecc71','#e74c3c'], ax=ax)
ax.set_title('Benign vs Malicious Traffic (Train)')
ax.set_ylabel('Record count')
plt.tight_layout(); plt.savefig('plots/01_benign_vs_malicious.png', dpi=110); plt.close()

# 2. Attack category distribution
fig, ax = plt.subplots(figsize=(8,5))
order = train['attack_cat'].value_counts().index
sns.countplot(y='attack_cat', data=train, order=order, ax=ax, palette='viridis')
ax.set_title('Attack Category Distribution (Train)')
plt.tight_layout(); plt.savefig('plots/02_attack_category_distribution.png', dpi=110); plt.close()

# 3. Class imbalance (log scale)
fig, ax = plt.subplots(figsize=(8,5))
counts = train['attack_cat'].value_counts()
ax.bar(counts.index, counts.values, color='#3498db')
ax.set_yscale('log')
ax.set_ylabel('Record count (log scale)')
ax.set_title('Class Imbalance Across Attack Categories')
plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.savefig('plots/03_class_imbalance_log.png', dpi=110); plt.close()

# 4. Feature distributions (a few key ones)
key_feats = ['dur','sbytes','dbytes','rate','sttl','dttl']
fig, axes = plt.subplots(2,3, figsize=(14,8))
for ax, feat in zip(axes.flat, key_feats):
    sns.histplot(np.log1p(train[feat]), bins=50, ax=ax, color='#9b59b6')
    ax.set_title(f'log1p({feat}) distribution')
plt.tight_layout(); plt.savefig('plots/04_feature_distributions.png', dpi=110); plt.close()

# 5. Correlation matrix (numeric features)
numeric_cols = train.select_dtypes(include=[np.number]).columns.drop(['id','label'])
corr = train[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(16,13))
sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax, cbar_kws={'shrink':0.7})
ax.set_title('Correlation Matrix — Numeric Flow Features')
plt.tight_layout(); plt.savefig('plots/05_correlation_matrix.png', dpi=110); plt.close()

# 6. Important feature viz - sttl by label (known strong signal in UNSW-NB15)
fig, ax = plt.subplots(figsize=(7,5))
sns.boxplot(x='label', y='sttl', data=train, ax=ax)
ax.set_xticklabels(['Benign','Malicious'])
ax.set_title('Source TTL (sttl) by Traffic Type')
plt.tight_layout(); plt.savefig('plots/06_sttl_by_label.png', dpi=110); plt.close()

# 7. Traffic stats table
traffic_stats = train[['dur','spkts','dpkts','sbytes','dbytes','rate']].describe().round(2)
traffic_stats.to_csv('plots/07_traffic_stats.csv')

log['phase5'] = {
    'benign_count_train': int((train['label']==0).sum()),
    'malicious_count_train': int((train['label']==1).sum()),
    'top_correlated_pairs': []
}
# top correlated feature pairs (excluding self-correlation)
corr_abs = corr.abs()
pairs = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool)).stack().sort_values(ascending=False).head(10)
log['phase5']['top_correlated_pairs'] = [f"{a}~{b}: {v:.2f}" for (a,b),v in pairs.items()]

print("PHASE 5 EDA plots saved.")

# ============================================================
# PHASE 6: FEATURE ENGINEERING
# ============================================================
# 'id' is a pure row identifier -> drop (no predictive meaning, would just let models
#   memorize row order / dataset construction artifacts).
# NOTE: this pre-extracted UNSW-NB15 feature set does NOT include raw source/destination
#   IP addresses or absolute timestamps (unlike raw CIC-IDS2017 flow captures), so that
#   classic IP/timestamp leakage risk does not apply to this feature set. Flagged, not
#   silently assumed.

# bucket rare protocols to avoid an explosion of one-hot columns from 133 raw categories
top_protos = train['proto'].value_counts().nlargest(10).index.tolist()
for df in (train, test):
    df['proto_bucket'] = df['proto'].where(df['proto'].isin(top_protos), 'other')

drop_cols = ['id', 'proto']  # proto replaced by proto_bucket
categorical_cols = ['proto_bucket', 'service', 'state']
target_cols = ['label', 'attack_cat']
feature_cols = [c for c in train.columns if c not in drop_cols + target_cols + categorical_cols]
numeric_cols_final = feature_cols  # all remaining are numeric

log['phase6'] = {
    'dropped_columns': drop_cols,
    'categorical_columns_encoded': categorical_cols,
    'proto_buckets_kept': top_protos,
    'numeric_feature_count': len(numeric_cols_final)
}

# mutual information (on a stratified sample of train, for speed) for feature selection insight
sample_idx = train.sample(n=min(40000, len(train)), random_state=RNG).index
mi_X = train.loc[sample_idx, numeric_cols_final]
mi_y = train.loc[sample_idx, 'label']
mi_scores = mutual_info_classif(mi_X, mi_y, random_state=RNG, discrete_features=False)
mi_series = pd.Series(mi_scores, index=numeric_cols_final).sort_values(ascending=False)
mi_series.head(20).to_csv('plots/08_mutual_information_top20.csv')
log['phase6']['top10_mutual_info_features'] = mi_series.head(10).round(4).to_dict()

print("PHASE 6 done:", json.dumps(log['phase6'], indent=2, default=str))

# ============================================================
# PHASE 7: TRAIN/TEST SPLIT
# ============================================================
# UNSW-NB15 ships with an official, pre-partitioned benchmark train/test split
# (175,341 / 82,332 rows). We use this official split rather than re-splitting
# 80/20 from scratch, since (a) it matches the standard evaluation protocol used
# in the literature for this exact dataset, and (b) re-shuffling and re-splitting
# would not remove the ~5.2% train/test overlap found in Phase 4 (that overlap is
# baked into the official partition itself) -- it is documented as a limitation
# instead. All preprocessing below is FIT ONLY on the training data.

X_train_raw = train[numeric_cols_final + categorical_cols]
X_test_raw = test[numeric_cols_final + categorical_cols]
y_train_bin = train['label'].values
y_test_bin = test['label'].values

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols_final),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)
feature_names_out = (numeric_cols_final +
                      list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)))

log['phase7'] = {'X_train_shape': list(X_train.shape), 'X_test_shape': list(X_test.shape)}
print("PHASE 7 done:", log['phase7'])

# ============================================================
# PHASE 8: CLASS IMBALANCE
# ============================================================
bin_counts = pd.Series(y_train_bin).value_counts()
imbalance_ratio = bin_counts[1] / bin_counts[0]
log['phase8'] = {
    'benign_train': int(bin_counts[0]), 'malicious_train': int(bin_counts[1]),
    'ratio_malicious_to_benign': round(float(imbalance_ratio), 2),
    'strategy': 'class_weight=balanced (binary); sample_weight balanced (multiclass/XGBoost). No SMOTE used -- ratio is moderate (~2:1) and network flow features are highly structured, so synthetic oversampling risks creating unrealistic flow patterns. class_weight is simpler, faster, and leaves the real data untouched.'
}
print("PHASE 8 done:", log['phase8'])

joblib.dump({'preprocessor': preprocessor, 'numeric_cols': numeric_cols_final,
             'categorical_cols': categorical_cols, 'top_protos': top_protos,
             'feature_names_out': feature_names_out}, 'models/preprocessor.joblib')

# checkpoint everything needed for next script
np.save('models/X_train.npy', X_train.toarray() if hasattr(X_train,'toarray') else X_train)
np.save('models/X_test.npy', X_test.toarray() if hasattr(X_test,'toarray') else X_test)
np.save('models/y_train_bin.npy', y_train_bin)
np.save('models/y_test_bin.npy', y_test_bin)
train[['attack_cat']].to_csv('models/y_train_attackcat.csv', index=False)
test[['attack_cat']].to_csv('models/y_test_attackcat.csv', index=False)
with open('models/log_phase4to8.json','w') as f:
    json.dump(log, f, indent=2, default=str)

print("\nCheckpoint saved. Phases 4-8 complete.")
