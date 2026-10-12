"""CICIoT2023 CSV prediction dashboard. Academic/demo use only."""

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Project root: parent directory of the pages folder
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "iot_threat_model.joblib"
META_PATH = ROOT / "iot_model_metadata.json"

MAX_UPLOAD_MB = 100
MAX_ROWS = 100_000

BENIGN_NAMES = {
    "benigntraffic",
    "benign",
    "normal",
    "normaltraffic",
}

# Page content
st.title("🌐 IoT Cyber Threat Detection")
st.caption(
    "CICIoT2023-based academic prototype • "
    "CSV prediction, not live Wi-Fi monitoring"
)

st.warning(
    "Predictions are model estimates, not confirmed incidents. "
    "This dashboard does not block devices or monitor the entire Wi-Fi network."
)

# Verify model files
if not MODEL_PATH.is_file() or not META_PATH.is_file():
    st.error(
        "IoT model files were not found. Expected both files in the "
        "repository root: iot_threat_model.joblib and iot_model_metadata.json."
    )
    st.stop()


@st.cache_resource
def load_bundle(path, mtime):
    # Load only model files created by your own trusted training script.
    return joblib.load(path)


@st.cache_data
def load_metadata(path, mtime):
    return json.loads(Path(path).read_text(encoding="utf-8"))


try:
    bundle = load_bundle(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

    metadata = load_metadata(
        str(META_PATH),
        META_PATH.stat().st_mtime,
    )

except Exception as exc:
    st.error(
        f"Could not load the IoT model: {type(exc).__name__}: {exc}"
    )
    st.stop()


# Model summary
st.subheader("📊 Model Overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Dataset rows used",
    f"{metadata['rows_used']:,}",
)

m2.metric(
    "Input features",
    str(metadata["feature_count"]),
)

m3.metric(
    "Attack/benign labels",
    str(metadata["class_count"]),
)

m4.metric(
    "Holdout accuracy",
    f"{metadata['multiclass_accuracy']:.1%}",
)

st.caption(
    "Holdout metrics come from the training run. "
    "They do not guarantee performance on new networks."
)

st.divider()

# CSV prediction
st.subheader("🔍 Analyze IoT Traffic CSV")

st.write(
    "Upload a CSV containing the same numeric feature columns "
    "used when training the IoT model."
)

upload = st.file_uploader(
    "Upload IoT traffic CSV",
    type=["csv"],
    key="iot_csv_upload",
)

if upload is not None:

    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"File is too large. Maximum allowed size is "
            f"{MAX_UPLOAD_MB} MB."
        )
        st.stop()

    try:
        df = pd.read_csv(
            upload,
            nrows=MAX_ROWS + 1,
            low_memory=False,
        )

    except Exception as exc:
        st.error(
            f"CSV could not be read: {type(exc).__name__}: {exc}"
        )
        st.stop()

    if len(df) > MAX_ROWS:
        st.error(
            f"CSV has more than {MAX_ROWS:,} rows. "
            "Please upload a smaller sample."
        )
        st.stop()

    if df.empty:
        st.error("The uploaded CSV contains no rows.")
        st.stop()

    # Check expected features
    expected = bundle["feature_columns"]

    has_label = "label" in df.columns

    actual_features = df.drop(
        columns=["label"],
        errors="ignore",
    )

    missing = [
        column
        for column in expected
        if column not in actual_features.columns
    ]

    extra = [
        column
        for column in actual_features.columns
        if column not in expected
    ]

    if missing:
        st.error(
            f"Required feature columns are missing ({len(missing)}): "
            f"{', '.join(missing[:12])}"
        )
        st.stop()

    if extra:
        st.info(
            f"Ignoring {len(extra)} extra column(s), such as: "
            f"{', '.join(extra[:8])}"
        )

    X = actual_features[expected].copy()

    # Validate numeric features
    bad = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if bad:
        st.error(
            "These feature columns must be numeric: "
            + ", ".join(bad[:12])
        )
        st.stop()

    # Handle infinite and missing values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    missing_counts = X.isna().sum()

    if missing_counts.sum():
        st.info(
            f"Missing or infinite values found in "
            f"{int((missing_counts > 0).sum())} feature(s). "
            "Training-set medians will be used where available."
        )

        medians = pd.Series(
            bundle.get("medians", {})
        )

        X = X.fillna(medians).fillna(0)

    # Prediction
    with st.spinner("Analyzing IoT traffic..."):

        try:
            predictions = bundle["model"].predict(X)

        except Exception as exc:
            st.error(
                f"Prediction failed: {type(exc).__name__}: {exc}"
            )
            st.stop()

        pred = pd.Series(
            predictions,
            index=df.index,
            name="predicted_label",
        )

        benign_names = {
            str(name).lower()
            for name in bundle.get(
                "benign_names",
                list(BENIGN_NAMES),
            )
        }

        benign_mask = (
            pred.astype(str)
            .str.lower()
            .isin(benign_names)
        )

        if hasattr(bundle["model"], "predict_proba"):
            try:
                probabilities = bundle["model"].predict_proba(X)
                confidence = probabilities.max(axis=1) * 100

            except Exception:
                confidence = np.full(
                    len(pred),
                    np.nan,
                )

        else:
            confidence = np.full(
                len(pred),
                np.nan,
            )

    # Results
    result = df.copy()

    result["predicted_label"] = pred.values

    result["traffic_class"] = np.where(
        benign_mask.values,
        "Benign",
        "Potential threat",
    )

    result["model_confidence_pct"] = np.round(
        confidence,
        2,
    )

    st.divider()
    st.subheader("📈 Analysis Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows analyzed",
        f"{len(result):,}",
    )

    c2.metric(
        "Predicted benign",
        f"{int(benign_mask.sum()):,}",
    )

    c3.metric(
        "Potential threats",
        f"{int((~benign_mask).sum()):,}",
    )

    # Distribution chart
    st.subheader("Predicted Label Distribution")

    label_counts = (
        result["predicted_label"]
        .value_counts()
        .head(20)
    )

    st.bar_chart(label_counts)

    # Optional comparison with uploaded labels
    if has_label:

        true_labels = (
            df["label"]
            .astype(str)
            .str.lower()
            .map(lambda value: re.sub(r"\s+", "", value))
        )

        pred_labels = (
            pred.astype(str)
            .str.lower()
            .map(lambda value: re.sub(r"\s+", "", value))
        )

        from sklearn.metrics import accuracy_score

        st.subheader("Uploaded Label Comparison")

        st.metric(
            "Accuracy on uploaded rows",
            f"{accuracy_score(true_labels, pred_labels):.1%}",
        )

        st.caption(
            "This comparison is meaningful only when the uploaded "
            "labels match the model's label format. It is not an "
            "independent test metric if these rows were used in training."
        )

    # Preview
    st.subheader("Prediction Preview")

    st.dataframe(
        result.head(200),
        use_container_width=True,
    )

    # Download results
    csv_bytes = result.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Predictions CSV",
        data=csv_bytes,
        file_name="iot_threat_predictions.csv",
        mime="text/csv",
    )


# Training evaluation
with st.expander("📋 Training Evaluation Details"):

    st.write(
        f"**Dataset:** {metadata['dataset']}"
    )

    st.write(
        f"**Training rows:** {metadata['train_rows']:,} "
        f"• **Test rows:** {metadata['test_rows']:,}"
    )

    st.write(
        f"**Macro F1:** {metadata['macro_f1']:.3f} "
        f"• **Weighted F1:** {metadata['weighted_f1']:.3f}"
    )

    st.write(
        "Per-class evaluation details are available in "
        "`iot_model_metadata.json`."
    )

    st.warning(
        "CICIoT2023 is a benchmark dataset. This dashboard analyzes "
        "uploaded CSV files; it does not collect live Wi-Fi traffic."
    )
