"""Advanced dashboard features: Explainable AI (SHAP), Risk Explorer, Clustering, and Alert History."""

from pathlib import Path
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def format_fraud_score(score) -> str:
    """Format fraud probability as a percentage without hiding small values.

    Examples:
        0.000001 -> 0.0001%
        0.0001   -> 0.0100%
        0.001    -> 0.1000%
        0.01     -> 1.0000%
        0.50     -> 50.0000%
        0.0      -> 0.0000%
    """
    if score is None or pd.isna(score):
        return "N/A"
    try:
        val = float(score)
        if 0.0 < val < 1e-6:
            return f"{val * 100:.6f}%"
        return f"{val * 100:.4f}%"
    except (ValueError, TypeError):
        return "N/A"


def calculate_risk_level(score, threshold, high_cutoff=None) -> str:
    """Compute risk level ('LOW', 'REVIEW', 'HIGH') based on score and threshold."""
    if score is None or pd.isna(score) or threshold is None or pd.isna(threshold):
        return "Unavailable"
    score_f = float(score)
    thresh_f = float(threshold)
    cutoff = max(thresh_f, 0.80) if high_cutoff is None else float(high_cutoff)

    if score_f < thresh_f:
        return "LOW"
    elif score_f >= cutoff:
        return "HIGH"
    else:
        return "REVIEW"


def get_prediction_result(actual, predicted) -> str | None:
    """Return the prediction classification result comparing actual Class to predicted class.

    Returns:
        'Correct prediction'       - True Negative (actual=0, pred=0)
        'False positive'           - False Positive (actual=0, pred=1)
        'Correct fraud detection'  - True Positive (actual=1, pred=1)
        'False negative'           - False Negative (actual=1, pred=0)
        None                       - If actual Class is missing or invalid
    """
    if actual is None or pd.isna(actual) or predicted is None or pd.isna(predicted):
        return None
    try:
        act = int(round(float(actual)))
        pred = int(round(float(predicted)))
    except (ValueError, TypeError):
        return None

    if act not in (0, 1) or pred not in (0, 1):
        return None

    if act == 0 and pred == 0:
        return "Correct prediction"
    elif act == 0 and pred == 1:
        return "False positive"
    elif act == 1 and pred == 1:
        return "Correct fraud detection"
    elif act == 1 and pred == 0:
        return "False negative"
    return None


def explain_transaction_shap(
    model,
    row_series: pd.Series,
    feature_columns: list[str],
    top_n: int = 8,
) -> dict | None:
    """Compute local SHAP feature contributions for a selected transaction."""
    try:
        import shap

        clean_cols = [c for c in feature_columns if c != "Class" and c in row_series.index]
        X_df = pd.DataFrame([row_series[clean_cols]])
        X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Handle Pipeline
        if hasattr(model, "named_steps"):
            tree_model = model.named_steps.get("model", model)
            imputer = model.named_steps.get("imputer")
            if imputer is not None:
                X_vals = imputer.transform(X_df)
            else:
                X_vals = X_df.values
        else:
            tree_model = model
            X_vals = X_df.values

        explainer = shap.TreeExplainer(tree_model)
        shap_vals = explainer.shap_values(X_vals)

        # Determine fraud class index
        fraud_class_idx = 1
        if hasattr(tree_model, "classes_") and 1 in tree_model.classes_:
            fraud_class_idx = list(tree_model.classes_).index(1)

        if isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                contribs = shap_vals[0, :, fraud_class_idx]
            else:
                contribs = shap_vals[0, :]
        elif isinstance(shap_vals, list):
            contribs = shap_vals[fraud_class_idx][0]
        else:
            return None

        # Base value
        if hasattr(explainer, "expected_value"):
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_val = float(explainer.expected_value[fraud_class_idx])
            else:
                base_val = float(explainer.expected_value)
        else:
            base_val = 0.5

        contrib_df = pd.DataFrame({
            "feature": clean_cols,
            "feature_value": [row_series[c] for c in clean_cols],
            "shap_value": contribs,
            "direction": np.where(contribs > 0, "Increases Fraud Risk", "Decreases Fraud Risk"),
        })

        # Sort by absolute impact
        contrib_df["abs_shap"] = contrib_df["shap_value"].abs()
        sorted_df = contrib_df.sort_values("abs_shap", ascending=False).drop(columns=["abs_shap"])

        top_positive = sorted_df[sorted_df["shap_value"] > 0].head(top_n)
        top_negative = sorted_df[sorted_df["shap_value"] < 0].head(top_n)

        return {
            "base_value": base_val,
            "top_features": sorted_df.head(top_n),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "all_contributions": sorted_df,
        }
    except Exception as exc:
        st.caption(f"SHAP explanation note: {exc}")
        return None


def render_interactive_visualizations(df):
    """Render interactive data distributions."""
    st.subheader("Interactive Transaction Visualizations")

    if df is None or df.empty:
        st.info("Upload a transaction CSV first.")
        return

    left, right = st.columns(2)

    with left:
        if "predicted_class" in df.columns:
            counts = (
                pd.to_numeric(df["predicted_class"], errors="coerce")
                .map({0: "Not flagged", 1: "Flagged"})
                .fillna("Unknown")
                .value_counts()
            )
            st.markdown("**Current Model Predictions**")
            st.bar_chart(counts)
        elif "Class" in df.columns:
            counts = (
                pd.to_numeric(df["Class"], errors="coerce")
                .map({0: "Legitimate", 1: "Fraud"})
                .fillna("Unknown")
                .value_counts()
            )
            st.markdown("**Actual Transaction Labels**")
            st.bar_chart(counts)

    with right:
        if "risk_level" in df.columns:
            st.markdown("**Current Risk-Level Distribution**")
            st.bar_chart(df["risk_level"].value_counts())
        elif "fraud_score" in df.columns:
            st.markdown("**Fraud Score Distribution**")
            st.bar_chart(df["fraud_score"])

    if "Amount" in df.columns:
        amounts = pd.to_numeric(df["Amount"], errors="coerce").dropna()
        if not amounts.empty:
            st.markdown("**Transaction Amount Distribution Filter**")
            max_amount = max(float(amounts.max()), 1.0)
            default_amount = min(max_amount * 0.25, max_amount)

            amount_limit = st.slider(
                "Filter amount ceiling for visualization ($)",
                min_value=0.0,
                max_value=max_amount,
                value=default_amount,
                key="amount_chart_limit",
            )

            filtered = df[pd.to_numeric(df["Amount"], errors="coerce").le(amount_limit)]
            if "predicted_class" in filtered.columns:
                counts = (
                    pd.to_numeric(filtered["predicted_class"], errors="coerce")
                    .map({0: "Not flagged", 1: "Flagged"})
                    .fillna("Unknown")
                    .value_counts()
                )
                st.bar_chart(counts)


def render_transaction_risk_explorer(
    df,
    threshold=None,
    model=None,
    feature_columns=None,
    fraud_class_index=None,
):
    """Detailed Individual Transaction Risk Explorer with Explainable AI (SHAP)."""
    st.subheader("Individual Transaction Risk Explorer")

    if df is None or df.empty:
        st.info("Upload a transaction CSV first.")
        return

    # Transaction row selection (1-based for users, converted to 0-based DataFrame index)
    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        row_number = st.number_input(
            "Select transaction row number (1-based, matches CSV row)",
            min_value=1,
            max_value=len(df),
            value=1,
            step=1,
            key="risk_explorer_row",
        )
    with c_sel2:
        st.metric("Total Uploaded Rows", f"{len(df):,}")

    position = int(row_number) - 1
    row = df.iloc[position]

    st.caption(
        f"Selected row **{position + 1}** of **{len(df):,}** "
        f"(DataFrame index {position}) in the uploaded dataset."
    )

    # 1. Resolve decision threshold
    if threshold is None:
        threshold = row.get("threshold_used", 0.5)
    try:
        threshold_val = float(threshold)
    except (ValueError, TypeError):
        threshold_val = 0.5

    # 2. Extract or compute fraud score
    score = pd.to_numeric(
        pd.Series([row.get("fraud_score", np.nan)]),
        errors="coerce",
    ).iloc[0]

    # Compute if score is missing but model is provided
    if pd.isna(score) and model is not None and feature_columns is not None:
        clean_features = [col for col in feature_columns if col != "Class" and col in row.index]
        X_single = pd.DataFrame([row[clean_features]])
        for col in X_single.columns:
            X_single[col] = pd.to_numeric(X_single[col], errors="coerce")
        X_single = X_single.replace([np.inf, -np.inf], np.nan)
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            f_idx = fraud_class_index if fraud_class_index is not None else (
                classes.index(1) if 1 in classes else 1
            )
            score = float(model.predict_proba(X_single)[0, f_idx])

    # 3. Model prediction & risk level calculation
    if pd.notna(score):
        pred_int = 1 if score >= threshold_val else 0
        prediction_label = "Flagged for review" if pred_int == 1 else "Not flagged"
        risk = calculate_risk_level(score, threshold_val)
    else:
        pred_val = pd.to_numeric(
            pd.Series([row.get("predicted_class", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(pred_val):
            pred_int = int(pred_val)
            prediction_label = "Flagged for review" if pred_int == 1 else "Not flagged"
        else:
            pred_int = None
            prediction_label = "N/A"
        risk = str(row.get("risk_level", "Unavailable"))

    # 4. Check for actual Class column in uploaded CSV
    has_class = "Class" in row.index and pd.notna(row["Class"])
    actual_class = None
    if has_class:
        try:
            val_c = float(row["Class"])
            if val_c in (0.0, 1.0):
                actual_class = int(val_c)
        except (ValueError, TypeError):
            actual_class = None

    prediction_result = (
        get_prediction_result(actual_class, pred_int)
        if actual_class is not None and pred_int is not None
        else None
    )

    # 5. Display Transaction Information Section
    st.markdown("### Transaction Risk Summary")

    row_amount = row.get("Amount", "N/A")
    formatted_amount = f"${float(row_amount):,.2f}" if pd.notna(pd.to_numeric(row_amount, errors="coerce")) else "N/A"

    if actual_class is not None:
        actual_label = "Fraud (Class = 1)" if actual_class == 1 else "Legitimate (Class = 0)"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Transaction", f"Row {position + 1}")
        c2.metric("Transaction Amount", formatted_amount)
        c3.metric("Fraud Score", format_fraud_score(score))
        c4.metric("Model Prediction", prediction_label)

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Risk Level", risk)
        c6.metric("Threshold Used", f"{threshold_val:.4f}")
        c7.metric("Actual Label", actual_label)
        c8.metric("Prediction Result", prediction_result if prediction_result else "N/A")

        if prediction_result == "Correct fraud detection":
            st.success(
                f"**{prediction_result} (True Positive)**: "
                "The model successfully detected and flagged this fraudulent transaction."
            )
        elif prediction_result == "Correct prediction":
            st.success(
                f"**{prediction_result} (True Negative)**: "
                "The model correctly identified this legitimate transaction as not flagged."
            )
        elif prediction_result == "False positive":
            st.warning(
                f"**{prediction_result}**: "
                "Legitimate transaction was flagged for human review (false alarm: score meets or exceeds threshold)."
            )
        elif prediction_result == "False negative":
            st.error(
                f"**{prediction_result}**: "
                "Fraudulent transaction was missed by the model (score is below threshold)."
            )
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Selected Transaction", f"Row {position + 1}")
        c2.metric("Transaction Amount", formatted_amount)
        c3.metric("Fraud Score", format_fraud_score(score))
        c4.metric("Model Prediction", prediction_label)
        c5.metric("Risk Level", risk)
        st.caption(f"Decision threshold used for prediction: **{threshold_val:.4f}**")

    # Risk level educational definitions
    st.caption(
        "**Risk Level Definitions:** "
        "• **HIGH**: Score ≥ 0.80 (elevated fraud risk). "
        "• **REVIEW**: Score between threshold and 0.80 (requires verification). "
        "• **LOW**: Score below threshold. **Note: LOW risk indicates lower probability, not an absolute guarantee of safety.**"
    )

    # 6. SHAP Explainability Section
    st.markdown("---")
    st.markdown("#### Explainable AI — Local Feature Contributions (SHAP)")
    st.caption(
        "Explains which specific transaction features pushed the fraud probability up or down for this selected transaction."
    )

    if model is not None and feature_columns is not None:
        shap_res = explain_transaction_shap(model, row, feature_columns, top_n=8)
        if shap_res is not None and not shap_res["top_features"].empty:
            sh_left, sh_right = st.columns([3, 2])

            with sh_left:
                st.markdown("**Top Feature Attribution Chart**")
                top_feat_df = shap_res["top_features"].iloc[::-1]  # ascending for horizontal bar chart

                fig, ax = plt.subplots(figsize=(7, 4.2))
                colors = ["#dc3545" if s > 0 else "#28a745" for s in top_feat_df["shap_value"]]
                bars = ax.barh(top_feat_df["feature"], top_feat_df["shap_value"], color=colors, height=0.6)
                ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
                ax.set_xlabel("SHAP Impact on Fraud Probability")
                ax.set_title("Feature Contribution (Red = Toward Fraud, Green = Toward Legitimate)", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with sh_right:
                st.markdown("**Top Contributing Features**")
                display_shap = shap_res["top_features"][["feature", "feature_value", "shap_value", "direction"]].copy()
                display_shap["feature_value"] = display_shap["feature_value"].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
                display_shap["shap_value"] = display_shap["shap_value"].apply(lambda v: f"{v:+.4f}")
                st.dataframe(display_shap, use_container_width=True, hide_index=True)

            st.caption(
                "**College Presentation Insight — Explainable AI:** "
                "Feature attributions explain the mathematical model's scoring mechanics for this specific transaction. "
                "They describe model behavior and do not constitute legal proof of fraudulent intent."
            )
        else:
            st.info("SHAP explanations are computing or model format requires manual review.")
    else:
        st.info("Model pipeline passed to explorer to enable live SHAP explanations.")

    # 7. Selected Transaction Details Table (preserves existing table & features)
    st.markdown("---")
    st.markdown("**Complete Transaction Feature Values**")
    st.dataframe(
        row.to_frame(name="Value"),
        use_container_width=True,
    )


def render_clustering_anomaly_explorer(df, max_sample=2000):
    """Exploratory K-Means and DBSCAN anomaly detection with cluster statistics."""
    st.subheader("K-Means and DBSCAN Anomaly Explorer")

    st.caption(
        "Exploratory unsupervised clustering only. Outliers and noise points represent statistical anomalies, "
        "not confirmed fraud cases. A representative sample is used for fast visualization."
    )

    if df is None or df.empty:
        st.info("Upload a transaction CSV first.")
        return

    excluded = {
        "Class", "predicted_class", "fraud_score", "risk_level",
        "review_status", "actual_class", "Prediction", "recorded_at",
        "source_row_number", "threshold_used", "upload_run_id", "alert_id",
    }

    candidates = [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(candidates) < 2:
        st.warning("At least two numeric transaction features are required for clustering.")
        return

    defaults = [c for c in ["Time", "Amount", "V1", "V2", "V3"] if c in candidates]
    if len(defaults) < 2:
        defaults = candidates[:min(5, len(candidates))]

    features = st.multiselect(
        "Choose numeric features for clustering",
        candidates,
        default=defaults,
        key="cluster_features_sel",
    )

    if len(features) < 2:
        st.info("Please select at least two numeric features.")
        return

    sample = (
        df.sample(n=min(max_sample, len(df)), random_state=42).copy()
        if len(df) > max_sample
        else df.copy()
    )

    raw = sample[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_mask = raw.notna().all(axis=1)
    raw = raw.loc[valid_mask]
    sample = sample.loc[valid_mask]

    if len(raw) < 10:
        st.warning("Not enough complete numeric records available for clustering.")
        return

    scaled = StandardScaler().fit_transform(raw)
    points = PCA(n_components=2, random_state=42).fit_transform(scaled)

    left, right = st.columns(2)

    with left:
        st.markdown("**K-Means Clustering**")
        k = st.slider("Number of clusters (K)", min_value=2, max_value=min(8, len(raw) - 1), value=3, key="kmeans_k_slider")
        try:
            km_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
            plot_km = pd.DataFrame({"PCA 1": points[:, 0], "PCA 2": points[:, 1], "Cluster": km_labels.astype(str)})
            st.scatter_chart(plot_km, x="PCA 1", y="PCA 2", color="Cluster")

            # Cluster statistics
            km_summary = pd.DataFrame({"Cluster": pd.Series(km_labels).value_counts().sort_index().index, "Transactions": pd.Series(km_labels).value_counts().sort_index().values})
            if "Amount" in sample.columns:
                km_amounts = sample.groupby(km_labels)["Amount"].agg(["mean", "max"]).reset_index()
                km_summary["Mean Amount"] = km_amounts["mean"].apply(lambda v: f"${v:,.2f}")
                km_summary["Max Amount"] = km_amounts["max"].apply(lambda v: f"${v:,.2f}")
            if "Class" in sample.columns:
                km_fraud = sample.groupby(km_labels)["Class"].agg(["sum", "mean"]).reset_index()
                km_summary["Actual Frauds"] = km_fraud["sum"].astype(int)
                km_summary["Fraud Rate"] = km_fraud["mean"].apply(lambda v: f"{v:.2%}")

            st.dataframe(km_summary, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"K-Means execution error: {exc}")

    with right:
        st.markdown("**DBSCAN Density-Based Anomaly Detection**")
        d1, d2 = st.columns(2)
        eps = d1.slider("DBSCAN eps", min_value=0.5, max_value=5.0, value=2.0, step=0.1, key="dbscan_eps_slider")
        min_samples = d2.slider("Min samples", min_value=3, max_value=30, value=10, key="dbscan_min_samples_slider")
        try:
            db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
            groups = np.where(db_labels == -1, "Noise / Anomaly", "Cluster " + db_labels.astype(str))
            plot_db = pd.DataFrame({"PCA 1": points[:, 0], "PCA 2": points[:, 1], "Group": groups})
            st.scatter_chart(plot_db, x="PCA 1", y="PCA 2", color="Group")

            noise_count = int((db_labels == -1).sum())
            noise_pct = (100.0 * noise_count / len(db_labels)) if len(db_labels) > 0 else 0.0

            db_m1, db_m2 = st.columns(2)
            db_m1.metric("Noise Points Detected", f"{noise_count:,}")
            db_m2.metric("Noise Ratio", f"{noise_pct:.2f}%")

            if "Class" in sample.columns:
                noise_fraud = int(((db_labels == -1) & (sample["Class"] == 1)).sum())
                st.caption(f"Among the {noise_count} detected noise points, **{noise_fraud}** are confirmed fraud transactions.")
        except Exception as exc:
            st.error(f"DBSCAN execution error: {exc}")


def render_fraud_alert_history(
    df,
    uploaded_filename="uploaded_transactions.csv",
    project_root=".",
    threshold=None,
):
    """Save, inspect, and export fraud alert history with filters and deduplication."""
    st.subheader("Fraud Alert History")
    st.caption(
        "Historical alerts are persisted separately from the current upload. "
        "Each alert is uniquely stamped with an Alert ID, Source File, Row Number, Score, Risk Level, and Decision Threshold."
    )

    history_path = Path(project_root) / "reports" / "fraud_alert_history.csv"
    legacy_path = Path(project_root) / "reports" / "fraud_alert_history_legacy_backup.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Legacy migration guard
    if history_path.exists():
        try:
            existing_cols = set(pd.read_csv(history_path, nrows=0).columns)
            if "upload_run_id" not in existing_cols:
                if not legacy_path.exists():
                    history_path.replace(legacy_path)
                else:
                    history_path.unlink()
        except Exception as exc:
            st.warning(f"Old alert history validation note: {exc}")

    # Save button for current upload
    if df is not None and not df.empty and "predicted_class" in df.columns:
        predictions = pd.to_numeric(df["predicted_class"], errors="coerce")
        flagged = df.loc[predictions.eq(1)].copy()
        st.write(f"Flagged transactions in current upload: **{len(flagged):,}**")

        if flagged.empty:
            st.info("No transactions are flagged at the current threshold.")
        elif st.button("Save current flagged transactions to alert history", key="save_fraud_alerts_btn"):
            saved = flagged.copy()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            saved.insert(0, "recorded_at", timestamp)
            saved.insert(1, "source_file", str(uploaded_filename))

            if "source_row_number" not in saved.columns:
                flagged_positions = [pos + 1 for pos, flag in enumerate(predictions) if flag == 1]
                saved["source_row_number"] = flagged_positions if len(flagged_positions) == len(saved) else (saved.index + 1)

            if "threshold_used" not in saved.columns or saved["threshold_used"].isna().all():
                saved["threshold_used"] = float(threshold) if threshold is not None else float("nan")

            if "upload_run_id" not in saved.columns:
                saved["upload_run_id"] = "run-" + hashlib.sha256(f"{uploaded_filename}_{timestamp}".encode()).hexdigest()[:12]

            # Deterministic Alert ID
            saved["alert_id"] = [
                f"ALT-{hashlib.sha256(f'{run_id}_{row_num}'.encode()).hexdigest()[:10].upper()}"
                for run_id, row_num in zip(saved["upload_run_id"], saved["source_row_number"])
            ]

            if history_path.exists():
                try:
                    previous = pd.read_csv(history_path)
                except Exception as exc:
                    st.error(f"Could not read existing alert history: {exc}")
                    return
            else:
                previous = pd.DataFrame()

            # Deduplication
            keys = ["upload_run_id", "source_row_number"]
            if not previous.empty and set(keys).issubset(previous.columns):
                prev_keys = previous[keys].astype(str).agg("|".join, axis=1)
                saved_keys = saved[keys].astype(str).agg("|".join, axis=1)
                saved = saved.loc[~saved_keys.isin(set(prev_keys))].copy()

            if saved.empty:
                st.info("These alerts were already saved for this upload and threshold.")
            else:
                combined = pd.concat([previous, saved], ignore_index=True, sort=False)
                combined.to_csv(history_path, index=False)
                st.success(f"Saved {len(saved):,} new alerts to {history_path.as_posix()}")

    # Display and filter historical alerts
    if history_path.exists():
        try:
            history = pd.read_csv(history_path)
            st.markdown("#### Saved Historical Alerts")
            st.metric("Total Historical Records", f"{len(history):,}")

            # Filters for alert history
            h_f1, h_f2, h_f3 = st.columns(3)
            hist_filtered = history.copy()

            if "risk_level" in hist_filtered.columns:
                r_opts = ["All"] + sorted(hist_filtered["risk_level"].dropna().unique().tolist())
                r_pick = h_f1.selectbox("Filter History by Risk Level", r_opts, key="hist_risk_filter")
                if r_pick != "All":
                    hist_filtered = hist_filtered[hist_filtered["risk_level"] == r_pick]

            if "Amount" in hist_filtered.columns:
                amt_h = pd.to_numeric(hist_filtered["Amount"], errors="coerce").dropna()
                if not amt_h.empty and amt_h.min() < amt_h.max():
                    h_bounds = h_f2.slider(
                        "History Amount Range ($)",
                        min_value=float(amt_h.min()),
                        max_value=float(amt_h.max()),
                        value=(float(amt_h.min()), float(amt_h.max())),
                        key="hist_amt_slider",
                    )
                    hist_filtered = hist_filtered[pd.to_numeric(hist_filtered["Amount"], errors="coerce").between(*h_bounds)]

            h_row_query = h_f3.number_input("Search History by Row Number", min_value=0, max_value=1_000_000, value=0, key="hist_row_search")
            if h_row_query > 0 and "source_row_number" in hist_filtered.columns:
                hist_filtered = hist_filtered[pd.to_numeric(hist_filtered["source_row_number"], errors="coerce") == h_row_query]

            st.markdown(f"**Showing {len(hist_filtered):,} matching historical alerts**")
            st.dataframe(hist_filtered.tail(200), use_container_width=True)

            d_c1, d_c2 = st.columns(2)
            with d_c1:
                st.download_button(
                    "Export Full Alert History (CSV)",
                    data=history.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_alert_history.csv",
                    mime="text/csv",
                    key="download_fraud_alert_history_btn",
                )
            with d_c2:
                if st.button("Clear Alert History", key="clear_fraud_alert_history_btn"):
                    history_path.unlink(missing_ok=True)
                    st.success("Alert history cleared.")
        except Exception as exc:
            st.error(f"Could not read alert history: {exc}")
    else:
        st.info("No alert history has been saved yet.")
