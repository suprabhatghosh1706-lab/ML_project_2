"""Streamlit dashboard for batch credit-card fraud scoring and Explainable AI."""

import json
import hashlib

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.common import MODEL_PATH, METRICS_PATH, validate_uploaded_dataframe

from src.dashboard_extras import (
    render_fraud_analytics,
    render_transaction_filters,
    render_confusion_matrix,
    render_model_comparison,
)

from src.advanced_features import (
    render_interactive_visualizations,
    render_transaction_risk_explorer,
    render_clustering_anomaly_explorer,
    render_fraud_alert_history,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection ML System",
    layout="wide",
)

st.title("Credit Card Fraud Detection ML System")

st.caption(
    "Enterprise-grade Machine Learning system for fraud detection, risk scoring, "
    "unsupervised anomaly exploration, and Explainable AI (SHAP)."
)


# --------------------------------------------------
# LOAD THE TRAINED MODEL (CACHED)
# --------------------------------------------------

@st.cache_resource
def load_model_bundle():
    """Load model artifact from disk with caching to ensure high responsiveness."""
    if not MODEL_PATH.exists():
        return None, "Trained model not found. First run: python -m src.train"
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle, None
    except Exception as exc:
        return None, f"Could not load the trained model: {exc}"


bundle, load_err = load_model_bundle()
if load_err or bundle is None:
    st.error(load_err if load_err else "Error loading model.")
    st.stop()

model = bundle["model"]
model_name = bundle.get("model_name", "Random Forest")
required_features = bundle["feature_columns"]

if "threshold" not in bundle:
    st.error(
        "This saved model has no matching decision threshold. "
        "Run `python -m src.train` to rebuild the model and reports."
    )
    st.stop()

default_threshold = float(bundle["threshold"])
recall_threshold = float(bundle.get("recall_focused_threshold", default_threshold))

if not hasattr(model, "predict_proba"):
    st.error("The saved model does not support predict_proba(). The dashboard requires a probability-capable classifier.")
    st.stop()

if "Class" in required_features:
    st.error("Target leakage detected: The saved model includes 'Class' as an input. Retrain without Class.")
    st.stop()

if not hasattr(model, "classes_") or 1 not in model.classes_:
    st.error("The trained model does not expose class 1 as the fraud class.")
    st.stop()

fraud_class_index = list(model.classes_).index(1)


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header("Model Configuration")

    st.markdown(f"**Active Classifier:** `{model_name}`")
    st.caption("Fitted with median imputation and class-weighting to combat extreme imbalance (0.173% fraud).")

    st.markdown("---")
    st.subheader("Decision Threshold Tuning")

    threshold = st.slider(
        "Fraud Decision Threshold",
        min_value=0.01,
        max_value=0.99,
        value=float(np.clip(default_threshold, 0.01, 0.99)),
        step=0.01,
        help="Transactions with fraud probability ≥ this threshold will be flagged for review.",
    )

    st.caption(
        f"• Default F1-Optimal Threshold: **{default_threshold:.4f}**\n"
        f"• Recall-Focused (F2) Threshold: **{recall_threshold:.4f}**"
    )

    st.info(
        f"**Active Threshold: {threshold:.4f}**\n\n"
        "**Tip:** Lower thresholds increase Recall (catch more fraudulent losses) at the expense of higher manual reviews (lower Precision)."
    )


# --------------------------------------------------
# DISPLAY SAVED MODEL EVALUATION METRICS
# --------------------------------------------------

if METRICS_PATH.exists():
    try:
        with open(METRICS_PATH, encoding="utf-8") as file:
            report = json.load(file)

        metrics = report.get("test_metrics", {})
        if report.get("selected_model") != model_name:
            st.warning(
                "The saved metrics were produced by a different model than the model currently loaded. "
                "Run `python -m src.train` to regenerate matching artifacts."
            )
            metrics = {}
        elif not np.isclose(
            float(report.get("decision_threshold", default_threshold)),
            default_threshold,
            atol=1e-8,
        ):
            st.warning("The saved metrics and model threshold differ. Run `python -m src.train` to synchronize.")
            metrics = {}

        st.subheader("Model Performance on Untouched Test Set (15% Split)")

        cols = st.columns(5)
        cols[0].metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}", help="Precision-Recall Area Under Curve — the primary metric for imbalanced fraud data.")
        cols[1].metric("Precision", f"{metrics.get('precision', 0):.2%}", help="Fraction of model alerts that are genuine fraud.")
        cols[2].metric("Recall (Sensitivity)", f"{metrics.get('recall', 0):.2%}", help="Fraction of actual fraud caught by the model.")
        cols[3].metric("F1-Score", f"{metrics.get('f1', 0):.4f}", help="Harmonic mean of Precision and Recall.")
        cols[4].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}", help="Area Under ROC Curve.")

        st.caption(
            f"Evaluated on {report.get('dataset_rows', 284807):,} transactions with {report.get('fraud_rows', 492):,} frauds. "
            f"Training Time: {report.get('training_time_sec', 'N/A')}s | Inference: {report.get('inference_time_sec', 'N/A')}s"
        )

    except Exception as exc:
        st.warning(f"Could not read saved evaluation metrics: {exc}")


# --------------------------------------------------
# UPLOAD TRANSACTION DATA
# --------------------------------------------------

st.divider()
st.subheader("Batch Transaction Scoring")

st.markdown(
    "Upload a transaction batch CSV containing the 30 numerical features (`Time`, `V1`–`V28`, `Amount`). "
    "An optional `Class` column may be included for ground-truth evaluation, but will **never** be used as a model input."
)

uploaded = st.file_uploader(
    "Choose transaction CSV to score",
    type=["csv"],
)


if uploaded is not None:
    try:
        incoming = pd.read_csv(uploaded)

        # Robust CSV Validation
        is_valid, val_err, X_valid_features = validate_uploaded_dataframe(incoming, required_features)
        if not is_valid:
            st.error(val_err)
            st.stop()

        # --------------------------------------------------
        # ONE SOURCE OF TRUTH FOR PREDICTIONS
        # --------------------------------------------------

        # Feature inputs strictly without Class
        feature_cols = [c for c in required_features if c != "Class"]
        X = incoming[feature_cols].copy()

        for column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        # Get continuous model score for class 1 (fraud)
        scores = model.predict_proba(X)[:, fraud_class_index]

        # Apply the configured threshold
        predictions = (scores >= threshold).astype(int)

        # Risk categories: HIGH (>= max(threshold, 0.80)), REVIEW (>= threshold), LOW (< threshold)
        high_cutoff = max(threshold, 0.80)
        risk_levels = np.where(
            predictions == 0,
            "LOW",
            np.where(scores >= high_cutoff, "HIGH", "REVIEW"),
        )

        review_statuses = np.where(
            predictions == 1,
            "Needs human review",
            "Not flagged by model",
        )

        # Build clean output dataframe
        generated_columns = [
            "fraud_score", "predicted_class", "risk_level",
            "review_status", "threshold_used", "source_row_number",
            "upload_run_id",
        ]
        output = incoming.drop(
            columns=[c for c in generated_columns if c in incoming.columns],
            errors="ignore",
        ).copy()

        # 1-based source row numbers
        output["source_row_number"] = np.arange(1, len(output) + 1)

        # Stable upload run ID for deduplication
        upload_bytes = uploaded.getvalue()
        run_sig = (
            hashlib.sha256(upload_bytes).hexdigest()
            + f"|{model_name}|{threshold:.6f}"
        )
        upload_run_id = hashlib.sha256(run_sig.encode("utf-8")).hexdigest()[:16]
        output["upload_run_id"] = upload_run_id

        output["fraud_score"] = scores
        output["predicted_class"] = predictions
        output["risk_level"] = risk_levels
        output["review_status"] = review_statuses
        output["threshold_used"] = threshold

        # --------------------------------------------------
        # SUMMARY CARDS
        # --------------------------------------------------

        total = len(output)
        flagged = int((output["predicted_class"] == 1).sum())
        not_flagged = total - flagged
        high_count = int((output["risk_level"] == "HIGH").sum())
        review_count = int((output["risk_level"] == "REVIEW").sum())
        low_count = int((output["risk_level"] == "LOW").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions Scored", f"{total:,}")
        c2.metric("Flagged for Review", f"{flagged:,}")
        c3.metric("Not Flagged", f"{not_flagged:,}")
        c4.metric("Flagged Rate", f"{(100.0 * flagged / total):.3f}%")

        st.caption(
            f"Risk Categorization: **HIGH** = {high_count:,} | "
            f"**REVIEW** = {review_count:,} | **LOW** = {low_count:,}"
        )

        # --------------------------------------------------
        # SCORED TRANSACTION RESULTS & DOWNLOAD
        # --------------------------------------------------

        st.subheader("Scored Transactions Preview")
        st.dataframe(output.head(200), use_container_width=True)

        st.download_button(
            "Download Complete Scored CSV",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="fraud_scored_predictions.csv",
            mime="text/csv",
        )

        # --------------------------------------------------
        # FRAUD ANALYTICS OVERVIEW
        # --------------------------------------------------

        st.divider()
        render_fraud_analytics(output)

        # --------------------------------------------------
        # SEARCH AND FILTER TRANSACTIONS
        # --------------------------------------------------

        st.divider()
        render_transaction_filters(output)

        # --------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------

        st.divider()
        if "Class" in incoming.columns:
            render_confusion_matrix(output)
        else:
            st.subheader("Confusion Matrix for Uploaded File")
            st.info(
                "Upload a dataset containing ground-truth 'Class' labels (0 = Legitimate, 1 = Fraud) "
                "to evaluate predictions and generate the confusion matrix."
            )

        # --------------------------------------------------
        # INTERACTIVE VISUALIZATIONS
        # --------------------------------------------------

        st.divider()
        render_interactive_visualizations(output)

        # --------------------------------------------------
        # INDIVIDUAL TRANSACTION RISK EXPLORER WITH SHAP
        # --------------------------------------------------

        st.divider()
        render_transaction_risk_explorer(
            output,
            threshold=threshold,
            model=model,
            feature_columns=required_features,
            fraud_class_index=fraud_class_index,
        )

        # --------------------------------------------------
        # K-MEANS AND DBSCAN ANOMALY EXPLORER
        # --------------------------------------------------

        st.divider()
        render_clustering_anomaly_explorer(output)

        # --------------------------------------------------
        # FRAUD ALERT HISTORY
        # --------------------------------------------------

        st.divider()
        render_fraud_alert_history(
            output,
            uploaded_filename=uploaded.name,
            project_root=".",
            threshold=threshold,
        )

    except Exception as exc:
        st.error(f"Could not score uploaded file: {exc}")


# --------------------------------------------------
# MODEL COMPARISON REPORT
# --------------------------------------------------

st.divider()
render_model_comparison(".")


# --------------------------------------------------
# PRESENTATION & EDUCATIONAL SUMMARY
# --------------------------------------------------

st.divider()
st.subheader("Key Concepts for Project Presentation")

st.markdown(
    "- **Class Imbalance Handling:** Real-world banking fraud is rare (~0.17%). Class weighting (`balanced_subsample`) adjusts the tree loss "
    "penalty inversely proportional to class frequencies without distorting validation/test distributions.\n"
    "- **Why PR-AUC Over ROC-AUC:** ROC-AUC includes True Negatives in the false alarm rate ($FP / (TN + FP)$), which inflates scores when $TN$ is massive. "
    "Precision-Recall AUC focuses strictly on fraud detection efficacy.\n"
    "- **Threshold Optimization:** The default threshold maximizes the $F_1$-score on held-out validation data. An operations manager can shift "
    "towards an $F_2$ threshold to catch more fraud at the expense of a few extra manual checks.\n"
    "- **Explainable AI (SHAP):** TreeExplainer calculates exact Shapley values showing which PCA features pushed each transaction's score.\n"
    "- **Responsible AI:** Predictions support human review; low risk is a probability metric, not an absolute guarantee of legitimacy."
)
