"""
threat_engine.py
Chunked scoring engine for the AI-Based Cyber Threat Detection Framework.

Reuses the already-trained artifacts (XGBoost binary model, Isolation Forest,
preprocessor, label encoder) and scores network-flow CSVs of ANY size by
reading them in chunks, so memory stays flat.

Input CSV must use UNSW-NB15 style flow columns (same as training), e.g.
dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, ... + proto, service, state.
Extra columns (id, label, attack_cat, src_ip, dst_ip, timestamp ...) are kept
for display but ignored by the models.
"""
import json
import os
from typing import Iterator, Tuple

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

SEVERITY = {
    'Normal': 0, 'Analysis': 40, 'Generic': 35, 'Reconnaissance': 40,
    'Fuzzers': 60, 'Backdoor': 60, 'Exploits': 80, 'Shellcode': 80,
    'DoS': 95, 'Worms': 95,
}
UNKNOWN_THREAT = 'Unclassified Threat'   # used when multiclass model file is absent
UNKNOWN_SEVERITY = 60


def risk_level(score: np.ndarray) -> np.ndarray:
    """Same thresholds as modeling.py: <=30 LOW, <=60 MEDIUM, <=80 HIGH, else CRITICAL."""
    return np.select([score <= 30, score <= 60, score <= 80],
                     ['LOW', 'MEDIUM', 'HIGH'], default='CRITICAL')


class ThreatEngine:
    def __init__(self, base_dir: str = BASE):
        self.base = base_dir
        meta = json.load(open(os.path.join(base_dir, 'model_metadata.json')))
        self.meta = meta
        self.model = joblib.load(os.path.join(base_dir, meta['best_model_file']))
        pre = joblib.load(os.path.join(base_dir, 'preprocessor.joblib'))
        self.preprocessor = pre['preprocessor']
        self.numeric_cols = pre['numeric_cols']
        self.categorical_cols = pre['categorical_cols']
        self.top_protos = set(pre['top_protos'])
        self.le = joblib.load(os.path.join(base_dir, 'label_encoder_attack_categories.joblib'))
        iso_path = os.path.join(base_dir, 'isolation_forest_anomaly_detector.joblib')
        self.iso = joblib.load(iso_path) if os.path.exists(iso_path) else None

        # Multiclass Random Forest is NOT in the repo zip (large file). Optional.
        mc_path = os.path.join(base_dir, meta.get('multiclass_model_file', ''))
        self.multiclass = joblib.load(mc_path) if os.path.isfile(mc_path) else None

        self.required_cols = self.numeric_cols + ['proto', 'service', 'state']

    # ---------------------------------------------------------------
    def check_columns(self, columns) -> Tuple[list, list]:
        cols = {c.strip().lower() for c in columns}
        missing = [c for c in self.required_cols if c not in cols]
        return missing, sorted(cols - set(self.required_cols))

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        X.columns = [c.strip().lower() for c in X.columns]
        for c in self.numeric_cols:
            if c not in X.columns:
                X[c] = 0.0
            X[c] = pd.to_numeric(X[c], errors='coerce')
        X[self.numeric_cols] = (X[self.numeric_cols]
                                .replace([np.inf, -np.inf], np.nan).fillna(0.0))
        for c in ('proto', 'service', 'state'):
            if c not in X.columns:
                X[c] = '-'
            X[c] = X[c].astype(str).str.strip()
        X['proto_bucket'] = X['proto'].where(X['proto'].isin(self.top_protos), 'other')
        return X[self.numeric_cols + self.categorical_cols]

    # ---------------------------------------------------------------
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score one chunk. Returns a frame aligned with df's rows."""
        X = self.preprocessor.transform(self._prepare(df))
        if hasattr(X, 'toarray'):
            X = X.toarray()

        proba = self.model.predict_proba(X)[:, 1]
        is_threat = proba >= 0.5
        confidence = np.where(is_threat, proba, 1 - proba) * 100

        if self.multiclass is not None:
            cat = self.le.inverse_transform(self.multiclass.predict(X))
            cat = np.where(is_threat, cat, 'Normal')
            # a flow flagged as threat by the binary model must not be labelled Normal
            cat = np.where(is_threat & (cat == 'Normal'), UNKNOWN_THREAT, cat)
        else:
            cat = np.where(is_threat, UNKNOWN_THREAT, 'Normal')

        severity = np.array([SEVERITY.get(c, UNKNOWN_SEVERITY) for c in cat])
        risk = (confidence / 100.0 * severity).astype(int)

        out = pd.DataFrame({
            'threat_category': cat,
            'confidence': confidence.round(1),
            'risk_score': risk,
            'risk_level': risk_level(risk),
            'threat_probability': proba.round(4),
        }, index=df.index)

        if self.iso is not None:
            out['anomaly_score'] = self.iso.score_samples(X).round(4)
            out['is_anomaly'] = self.iso.predict(X) == -1
        else:
            out['anomaly_score'] = 0.0
            out['is_anomaly'] = False
        return out

    # ---------------------------------------------------------------
    def stream_csv(self, source, chunksize: int = 20000) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Yield (raw_chunk, scored_chunk) for a path or file-like CSV, any size."""
        for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False,
                                 skipinitialspace=True):
            chunk.columns = [c.strip().lower() for c in chunk.columns]
            yield chunk, self.score(chunk)
