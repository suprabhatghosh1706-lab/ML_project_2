"""Dashboard analytics, filtering, confusion matrix, and model comparison views."""

from pathlib import Path
import json
import pandas as pd
import streamlit as st


def _col(df, names):
    """Find a dataframe column without depending on capitalization."""
    lookup = {str(column).lower(): column for column in df.columns}
    return next(
        (lookup[name.lower()] for name in names if name.lower() in lookup),
        None,
    )


def render_fraud_analytics(df):
    """Render comprehensive fraud analytics cards and distributions."""
    st.subheader("Fraud Analytics Overview")

    if df is None or df.empty:
        st.info("No scored transactions are available.")
        return

    pred_col = _col(df, ["predicted_class", "prediction", "predicted_label"])
    risk_col = _col(df, ["risk_level", "risk", "risk_category"])
    amount_col = _col(df, ["Amount", "amount", "transaction_amount"])
    score_col = _col(df, ["fraud_score", "fraud_probability", "score"])
    class_col = _col(df, ["Class", "actual_class", "actual_label", "target"])

    total = len(df)

    if pred_col:
        predictions = pd.to_numeric(df[pred_col], errors="coerce")
        flagged_mask = predictions.eq(1)
        flagged = int(flagged_mask.sum())
    else:
        flagged_mask = pd.Series(False, index=df.index)
        flagged = 0

    not_flagged = total - flagged
    flagged_pct = (100.0 * flagged / total) if total > 0 else 0.0

    # 1. Primary volume metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Flagged for Review", f"{flagged:,}")
    c3.metric("Not Flagged", f"{not_flagged:,}")
    c4.metric("Flagged Rate", f"{flagged_pct:.3f}%")

    # 2. Risk category breakdown
    if risk_col:
        risk_series = df[risk_col].fillna("Unknown").astype(str)
        high_cnt = int((risk_series == "HIGH").sum())
        rev_cnt = int((risk_series == "REVIEW").sum())
        low_cnt = int((risk_series == "LOW").sum())

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("HIGH Risk", f"{high_cnt:,}", help="High predicted fraud risk (score ≥ 0.80)")
        r2.metric("REVIEW Required", f"{rev_cnt:,}", help="Flagged transactions between threshold and 0.80")
        r3.metric("LOW Risk", f"{low_cnt:,}", help="Transactions below threshold. Note: LOW does not guarantee legitimate.")
        if amount_col and pred_col:
            flagged_amount = pd.to_numeric(
                df.loc[flagged_mask, amount_col],
                errors="coerce",
            ).fillna(0).sum()
            r4.metric("Flagged Volume", f"${flagged_amount:,.2f}")
        elif score_col:
            avg_score = pd.to_numeric(df[score_col], errors="coerce").mean()
            r4.metric("Average Fraud Score", f"{avg_score:.4f}" if pd.notna(avg_score) else "N/A")

    if amount_col and pred_col:
        st.caption(
            "**Note on Flagged Volume:** This represents the total transaction value flagged by the model for human inspection, "
            "not confirmed fraud losses."
        )

    # 3. Ground Truth evaluation metrics if Class is available
    has_ground_truth = class_col and pd.to_numeric(df[class_col], errors="coerce").dropna().isin([0, 1]).any()
    if has_ground_truth and pred_col:
        st.markdown("#### Ground-Truth Fraud Detection Performance")
        y = pd.to_numeric(df[class_col], errors="coerce")
        p = pd.to_numeric(df[pred_col], errors="coerce")
        valid = y.isin([0, 1]) & p.isin([0, 1])

        if valid.any():
            y_v = y[valid].astype(int)
            p_v = p[valid].astype(int)

            actual_fraud = int((y_v == 1).sum())
            actual_legit = int((y_v == 0).sum())
            tp = int(((y_v == 1) & (p_v == 1)).sum())
            fn = int(((y_v == 1) & (p_v == 0)).sum())
            fp = int(((y_v == 0) & (p_v == 1)).sum())
            tn = int(((y_v == 0) & (p_v == 0)).sum())

            recall = (tp / actual_fraud * 100.0) if actual_fraud > 0 else 0.0
            precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Actual Frauds in File", f"{actual_fraud:,}")
            g2.metric("Detected Frauds (TP)", f"{tp:,}", delta=f"{recall:.1f}% recall")
            g3.metric("Missed Frauds (FN)", f"{fn:,}", delta=f"-{fn}" if fn > 0 else "0", delta_color="inverse")
            g4.metric("False Alarms (FP)", f"{fp:,}", delta=f"{precision:.1f}% precision")

    # 4. Distribution charts
    col_left, col_right = st.columns(2)
    with col_left:
        if risk_col:
            st.markdown("**Risk-Level Distribution**")
            counts = df[risk_col].fillna("Unknown").astype(str).value_counts()
            st.bar_chart(counts)
        elif pred_col:
            st.markdown("**Model Decision Distribution**")
            st.bar_chart(predictions.map({0: "Not flagged", 1: "Flagged"}).value_counts())

    with col_right:
        if amount_col:
            st.markdown("**Transaction Amounts in Scored File**")
            amounts = pd.to_numeric(df[amount_col], errors="coerce").dropna()
            if not amounts.empty:
                amt_summary = pd.DataFrame({
                    "Statistic": ["Mean Amount", "Median Amount", "Max Amount", "Total Volume"],
                    "Value": [
                        f"${amounts.mean():,.2f}",
                        f"${amounts.median():,.2f}",
                        f"${amounts.max():,.2f}",
                        f"${amounts.sum():,.2f}",
                    ]
                })
                st.dataframe(amt_summary, use_container_width=True, hide_index=True)


def render_transaction_filters(df):
    """Interactive multi-criteria search and filtering for transactions."""
    st.subheader("Search and Filter Transactions")

    if df is None or df.empty:
        st.info("No scored transactions are available.")
        return

    filtered = df.copy()

    risk_col = _col(filtered, ["risk_level", "risk", "risk_category"])
    pred_col = _col(filtered, ["predicted_class", "prediction", "predicted_label"])
    amount_col = _col(filtered, ["Amount", "amount", "transaction_amount"])
    class_col = _col(filtered, ["Class", "actual_class", "actual_label", "target"])
    row_col = _col(filtered, ["source_row_number", "row_number"])

    # Search bar & row jumper
    top_c1, top_c2 = st.columns([2, 1])
    with top_c1:
        search_row = st.number_input(
            "Jump to specific transaction row number (1-based)",
            min_value=0,
            max_value=len(df),
            value=0,
            step=1,
            help="Set to 0 to view all rows, or enter a specific row number to isolate it.",
            key="filter_search_row",
        )
    with top_c2:
        amount_query = st.text_input(
            "Search minimum amount ($)",
            value="",
            placeholder="e.g. 500",
            help="Show transactions with amount greater than or equal to this value.",
            key="filter_min_amount_input",
        )

    if search_row > 0:
        if row_col and row_col in filtered.columns:
            filtered = filtered[pd.to_numeric(filtered[row_col], errors="coerce") == search_row]
        else:
            pos = search_row - 1
            if 0 <= pos < len(filtered):
                filtered = filtered.iloc[[pos]]

    if amount_query.strip():
        try:
            min_val = float(amount_query.strip())
            if amount_col:
                filtered = filtered[pd.to_numeric(filtered[amount_col], errors="coerce") >= min_val]
        except ValueError:
            pass

    # Multi-facet filters
    f1, f2, f3 = st.columns(3)

    if risk_col:
        risk_opts = ["All"] + sorted(filtered[risk_col].fillna("Unknown").astype(str).unique().tolist())
        risk_choice = f1.selectbox("Risk Level", risk_opts, key="filter_risk_choice")
        if risk_choice != "All":
            filtered = filtered[filtered[risk_col].fillna("Unknown").astype(str) == risk_choice]

    if pred_col:
        pred_choice = f2.selectbox("Model Prediction", ["All", "Flagged (1)", "Not flagged (0)"], key="filter_pred_choice")
        if pred_choice == "Flagged (1)":
            filtered = filtered[pd.to_numeric(filtered[pred_col], errors="coerce") == 1]
        elif pred_choice == "Not flagged (0)":
            filtered = filtered[pd.to_numeric(filtered[pred_col], errors="coerce") == 0]

    if class_col and pd.to_numeric(df[class_col], errors="coerce").dropna().isin([0, 1]).any():
        class_choice = f3.selectbox("Actual Label (Ground Truth)", ["All", "Actual Fraud (1)", "Actual Legitimate (0)"], key="filter_class_choice")
        if class_choice == "Actual Fraud (1)":
            filtered = filtered[pd.to_numeric(filtered[class_col], errors="coerce") == 1]
        elif class_choice == "Actual Legitimate (0)":
            filtered = filtered[pd.to_numeric(filtered[class_col], errors="coerce") == 0]
    elif amount_col:
        amt_series = pd.to_numeric(filtered[amount_col], errors="coerce").dropna()
        if not amt_series.empty and amt_series.min() < amt_series.max():
            bounds = f3.slider(
                "Amount range ($)",
                min_value=float(amt_series.min()),
                max_value=float(amt_series.max()),
                value=(float(amt_series.min()), float(amt_series.max())),
                key="filter_amount_slider",
            )
            filtered = filtered[pd.to_numeric(filtered[amount_col], errors="coerce").between(*bounds)]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} transactions matching active criteria**")

    # Render results table
    st.dataframe(
        filtered.head(1000),
        use_container_width=True,
    )

    if len(filtered) > 1000:
        st.info("Showing the first 1,000 filtered rows for fast response. Use the download button below to export full results.")


def render_confusion_matrix(df):
    """Render interactive confusion matrix heatmap and metrics for uploaded transactions."""
    st.subheader("Confusion Matrix for Uploaded File")

    if df is None or df.empty:
        st.info("No scored transactions are available.")
        return

    actual_col = _col(df, ["Class", "actual_class", "actual_label", "target"])
    pred_col = _col(df, ["predicted_class", "prediction", "predicted_label"])

    if not actual_col or not pred_col:
        st.info("Evaluation requires both the actual Class label column and model predictions.")
        return

    y = pd.to_numeric(df[actual_col], errors="coerce")
    p = pd.to_numeric(df[pred_col], errors="coerce")
    valid = y.isin([0, 1]) & p.isin([0, 1])

    if not valid.any():
        st.warning("No valid binary ground-truth labels (0 or 1) found in the uploaded file.")
        return

    y_val = y[valid].astype(int)
    p_val = p[valid].astype(int)

    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
    )

    cm = confusion_matrix(y_val, p_val, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_val, p_val, zero_division=0)
    recall = recall_score(y_val, p_val, zero_division=0)
    f1 = f1_score(y_val, p_val, zero_division=0)
    accuracy = accuracy_score(y_val, p_val)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Primary metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f"{precision:.4%}", help="Of all flagged transactions, what fraction was genuine fraud?")
    m2.metric("Recall (Sensitivity)", f"{recall:.4%}", help="Of all actual frauds, what fraction did the model catch?")
    m3.metric("F1-Score", f"{f1:.4f}", help="Harmonic mean of precision and recall.")
    m4.metric("Specificity", f"{specificity:.4%}", help="Of all legitimate transactions, what fraction was correctly unflagged?")
    m5.metric("Accuracy", f"{accuracy:.4%}", help="Overall accuracy (can be deceptive under high class imbalance).")

    # Confusion Matrix Table with human explanations
    st.markdown("#### Confusion Matrix Breakdown")
    cm_display = pd.DataFrame(
        [
            [f"TN: {tn:,} (Legitimate correctly unflagged)", f"FP: {fp:,} (False alarm: Legitimate flagged)"],
            [f"FN: {fn:,} (Missed fraud: Undetected)", f"TP: {tp:,} (Fraud correctly detected)"],
        ],
        index=["Actual Legitimate (0)", "Actual Fraud (1)"],
        columns=["Predicted Legitimate (0)", "Predicted Fraud (1)"],
    )
    st.dataframe(cm_display, use_container_width=True)

    # Educational alert on metric interpretation
    st.info(
        "**College Presentation Insight — Why Accuracy is Deceptive for Fraud Detection:**\n\n"
        f"In credit card transaction data, ~99.8% of transactions are legitimate. A trivial 'dummy' model predicting 0 for every transaction "
        f"would achieve an impressive **99.8% Accuracy** while catching **0% of frauds (0% Recall)**. "
        f"Therefore, banking systems rely on **PR-AUC, Recall (catching loss), and Precision (investigation workload)** rather than raw accuracy."
    )


def render_model_comparison(project_root="."):
    """Render comparison of candidate ML models and final held-out test evaluation."""
    st.subheader("ML Algorithm Comparison & Evaluation")

    root = Path(project_root)
    comp_path = root / "reports" / "model_comparison.csv"
    metrics_path = root / "reports" / "model_metrics.json"

    if not comp_path.exists():
        st.info("No reports/model_comparison.csv found. Run `python -m src.train` to generate the comparison report.")
        return

    try:
        results = pd.read_csv(comp_path)
    except Exception as exc:
        st.error(f"Could not read model comparison report: {exc}")
        return

    if results.empty:
        st.info("Model comparison report is empty.")
        return

    st.markdown("#### Validation Set Model Comparison (Stratified 70/15/15 Split)")
    st.caption("Models were trained on the 70% training split and evaluated strictly on the 15% validation split to prevent data leakage.")

    # Reorder columns for presentation clarity
    col_order = [
        "model",
        "validation_pr_auc",
        "validation_f1",
        "validation_recall",
        "validation_precision",
        "validation_roc_auc",
        "validation_accuracy",
        "validation_threshold",
        "training_time_sec",
        "inference_time_sec",
    ]
    present_cols = [c for c in col_order if c in results.columns]
    display_results = results[present_cols].copy()

    st.dataframe(display_results, use_container_width=True, hide_index=True)

    # Chart comparison
    metric_cols = [c for c in ["validation_pr_auc", "validation_f1", "validation_recall", "validation_precision"] if c in results.columns]
    if "model" in results.columns and metric_cols:
        st.markdown("**Validation Metrics Comparison by Algorithm**")
        chart_data = results.set_index("model")[metric_cols]
        st.bar_chart(chart_data)

    # Display final held-out test evaluation
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                report = json.load(f)

            selected = report.get("selected_model", "Random Forest")
            test_m = report.get("test_metrics", {})
            recall_m = report.get("recall_test_metrics", {})
            thresh = report.get("decision_threshold", "N/A")
            rec_thresh = report.get("recall_focused_threshold")

            st.markdown(f"#### Final Evaluation on Untouched Test Set (Selected Model: {selected})")
            st.caption("Evaluated strictly once on the 15% test set after all model and threshold selections were finalized.")

            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("PR-AUC", f"{test_m.get('pr_auc', 0):.4f}")
            t2.metric("Precision", f"{test_m.get('precision', 0):.2%}")
            t3.metric("Recall", f"{test_m.get('recall', 0):.2%}")
            t4.metric("F1-Score", f"{test_m.get('f1', 0):.4f}")
            t5.metric("ROC-AUC", f"{test_m.get('roc_auc', 0):.4f}")

            # Test confusion matrix
            tn = test_m.get("tn", 0)
            fp = test_m.get("fp", 0)
            fn = test_m.get("fn", 0)
            tp = test_m.get("tp", 0)

            st.markdown(f"**Test Confusion Matrix (Threshold = {thresh:.4f})**")
            test_cm_df = pd.DataFrame(
                [
                    [f"TN: {tn:,}", f"FP: {fp:,}"],
                    [f"FN: {fn:,}", f"TP: {tp:,}"],
                ],
                index=["Actual Legitimate (0)", "Actual Fraud (1)"],
                columns=["Predicted Legitimate (0)", "Predicted Fraud (1)"],
            )
            st.dataframe(test_cm_df, use_container_width=True)

            # Sensitivity analysis: Recall-focused threshold vs F1-optimal threshold
            if rec_thresh is not None and recall_m:
                with st.expander("Decision Threshold Sensitivity Analysis (F1 vs Recall-Focused)"):
                    st.markdown(
                        "In fraud operations, bank managers choose between maximizing the **F1-Score** (balanced balance) "
                        "and a **Recall-Focused Threshold** ($F_2$-Score, catching more fraud at the cost of a few extra manual reviews)."
                    )
                    sens_df = pd.DataFrame([
                        {
                            "Strategy": "F1-Maximizing (Standard)",
                            "Threshold": f"{thresh:.4f}",
                            "Recall": f"{test_m.get('recall', 0):.2%}",
                            "Precision": f"{test_m.get('precision', 0):.2%}",
                            "F1": f"{test_m.get('f1', 0):.4f}",
                            "Caught Frauds (TP)": f"{tp}",
                            "Missed Frauds (FN)": f"{fn}",
                            "False Alarms (FP)": f"{fp}",
                        },
                        {
                            "Strategy": "Recall-Focused (F2 Beta=2)",
                            "Threshold": f"{rec_thresh:.4f}",
                            "Recall": f"{recall_m.get('recall', 0):.2%}",
                            "Precision": f"{recall_m.get('precision', 0):.2%}",
                            "F1": f"{recall_m.get('f1', 0):.4f}",
                            "Caught Frauds (TP)": f"{recall_m.get('tp', 0)}",
                            "Missed Frauds (FN)": f"{recall_m.get('fn', 0)}",
                            "False Alarms (FP)": f"{recall_m.get('fp', 0)}",
                        },
                    ])
                    st.dataframe(sens_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Could not load final test metrics: {exc}")
