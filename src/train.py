"""Train and compare fraud classifiers without data leakage."""
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix, precision_recall_curve
)

from src.models import (
    create_dummy_model,
    create_logistic_regression_model,
    create_random_forest_model,
    create_extra_trees_model,
)

from src.common import (
    load_dataset, feature_columns, ARTIFACT_DIR, REPORT_DIR,
    MODEL_PATH, METRICS_PATH, COMPARISON_PATH
)

RANDOM_STATE = 42

def safe_auc(metric_fn, y_true, scores):
    try:
        return float(metric_fn(y_true, scores))
    except ValueError:
        return float("nan")

def metrics_at_threshold(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": safe_auc(roc_auc_score, y_true, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def choose_threshold(y_true, scores):
    """Choose an F1-maximising threshold on validation data, not the test set."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    # Avoid thresholds that flag virtually every transaction where possible.
    best_idx = int(np.nanargmax(f1_values))
    return float(np.clip(thresholds[best_idx], 0.0, 1.0))

def choose_recall_focused_threshold(y_true, scores, beta=2.0):
    """Choose a threshold that weights recall higher using the F_beta score (default beta=2)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision[:-1] * recall[:-1]) / (
        beta_sq * precision[:-1] + recall[:-1] + 1e-12
    )
    best_idx = int(np.nanargmax(f_beta))
    return float(np.clip(thresholds[best_idx], 0.0, 1.0))

def main():
    df = load_dataset()
    features = feature_columns(df)
    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = df["Class"].astype(int)

    # Stratified 70/15/15 split. Test data stays untouched until final evaluation.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    models = {
        "Dummy": create_dummy_model(),
        "Logistic Regression": create_logistic_regression_model(random_state=RANDOM_STATE),
        "Random Forest": create_random_forest_model(random_state=RANDOM_STATE),
        "Extra Trees": create_extra_trees_model(random_state=RANDOM_STATE),
    }

    comparison = []
    trained = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
        except Exception as exc:
            raise RuntimeError(f"Failed to train model '{name}': {exc}") from exc
        train_time_sec = round(time.time() - t0, 3)

        # Derive the fraud class index from model.classes_ to avoid
        # hardcoding column index 1 (which can fail if label ordering differs).
        t_infer0 = time.time()
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            fraud_idx = classes.index(1) if 1 in classes else 1
            valid_scores = model.predict_proba(X_valid)[:, fraud_idx]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X_valid)
            valid_scores = 1.0 / (1.0 + np.exp(-raw))
        else:
            warnings.warn(
                f"{name} supports neither predict_proba nor decision_function. "
                "Using binary predictions for scoring (PR-AUC will be unreliable)."
            )
            valid_scores = model.predict(X_valid).astype(float)
        inference_time_sec = round(time.time() - t_infer0, 4)

        threshold = choose_threshold(y_valid.to_numpy(), valid_scores)
        valid_metrics = metrics_at_threshold(y_valid.to_numpy(), valid_scores, threshold)
        comparison.append({
            "model": name,
            "validation_pr_auc": valid_metrics["pr_auc"],
            "validation_f1": valid_metrics["f1"],
            "validation_recall": valid_metrics["recall"],
            "validation_precision": valid_metrics["precision"],
            "validation_roc_auc": valid_metrics["roc_auc"],
            "validation_accuracy": valid_metrics["accuracy"],
            "validation_threshold": threshold,
            "training_time_sec": train_time_sec,
            "inference_time_sec": inference_time_sec,
        })
        trained[name] = {
            "model": model,
            "threshold": threshold,
            "valid_metrics": valid_metrics,
            "valid_scores": valid_scores,
            "train_time_sec": train_time_sec,
            "inference_time_sec": inference_time_sec,
        }

    comparison_df = pd.DataFrame(comparison).sort_values(
        "validation_pr_auc", ascending=False
    )
    best_name = str(comparison_df.iloc[0]["model"])
    best = trained[best_name]
    best_model = best["model"]
    threshold = float(best["threshold"])
    best_valid_metrics = best["valid_metrics"]

    # Derive fraud class index for the best model.
    if hasattr(best_model, "classes_"):
        fraud_class_idx = (
            list(best_model.classes_).index(1)
            if 1 in list(best_model.classes_)
            else 1
        )
    else:
        fraud_class_idx = 1

    # Evaluate the selected model ONCE on the untouched test set.
    t_test0 = time.time()
    if hasattr(best_model, "predict_proba"):
        test_scores = best_model.predict_proba(X_test)[:, fraud_class_idx]
    elif hasattr(best_model, "decision_function"):
        raw_test = best_model.decision_function(X_test)
        test_scores = 1.0 / (1.0 + np.exp(-raw_test))
    else:
        warnings.warn(
            "Best model does not support predict_proba or decision_function. "
            "Using binary predictions for test evaluation."
        )
        test_scores = best_model.predict(X_test).astype(float)
    test_eval_time_sec = round(time.time() - t_test0, 4)

    test_metrics = metrics_at_threshold(y_test.to_numpy(), test_scores, threshold)

    # Optional recall-focused evaluation (F2-maximizing threshold on validation)
    recall_threshold = choose_recall_focused_threshold(y_valid.to_numpy(), best["valid_scores"])
    recall_test_metrics = metrics_at_threshold(y_test.to_numpy(), test_scores, recall_threshold)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "threshold": threshold,
        "recall_focused_threshold": recall_threshold,
        "feature_columns": features,
        "fraud_class_label": 1,
        "fraud_class_index": fraud_class_idx,
        "validation_metrics": best_valid_metrics,
        "test_metrics": test_metrics,
        "recall_test_metrics": recall_test_metrics,
        "random_state": RANDOM_STATE,
        "model_version": "2.0",
        "training_time_sec": best["train_time_sec"],
        "inference_time_sec": best["inference_time_sec"],
        "comparison_records": comparison_df.to_dict(orient="records"),
    }, MODEL_PATH)

    comparison_df.to_csv(COMPARISON_PATH, index=False)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "selected_model": best_name,
            "decision_threshold": threshold,
            "recall_focused_threshold": recall_threshold,
            "test_metrics": test_metrics,
            "recall_test_metrics": recall_test_metrics,
            "dataset_rows": int(len(df)),
            "fraud_rows": int((y == 1).sum()),
            "legitimate_rows": int((y == 0).sum()),
            "training_time_sec": best["train_time_sec"],
            "inference_time_sec": best["inference_time_sec"],
            "features": features,
            "note": "Metrics are from this run; do not report them before training.",
        }, f, indent=2)

    print("\nValidation comparison (models sorted by PR-AUC):")
    print(comparison_df.to_string(index=False))
    print("\nSelected model:", best_name)
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    print("\nSaved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved comparison:", COMPARISON_PATH)

if __name__ == "__main__":
    main()
