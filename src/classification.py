"""Optional multi-algorithm comparison script.

This script compares six classifiers and saves a comparison-only model.
It does NOT replace the dashboard model (artifacts/fraud_model.joblib).
The dashboard model is only produced by:  python -m src.train

Key correctness guarantees
--------------------------
* Models are trained on a training subset; metrics are computed on a
  held-out validation set.  The test set is NOT used here.
* PR-AUC is always computed from continuous probability / decision scores,
  never from binary predictions.
* The fraud class index is derived from model.classes_ so that
  predict_proba[:,1] is correct even if the label ordering differs.
* SLP / Perceptron does not support predict_proba(); its decision_function
  output is passed through sigmoid to produce calibration-free probability
  estimates suitable for ranking metrics such as PR-AUC.
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, classification_report,
    precision_recall_curve,
)
import joblib

from src.models import (
    create_logistic_regression_model,
    create_knn_model,
    create_naive_bayes_model,
    create_perceptron_model,
    create_mlp_model,
    create_random_forest_model,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

RANDOM_STATE = 42


def _sigmoid(x):
    """Numerically stable sigmoid for converting decision scores to [0, 1]."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _get_scores(pipeline, X, fraud_label=1):
    """Return a continuous fraud score array for X.

    Priority order:
    1. predict_proba  – use the column matching fraud_label.
    2. decision_function – pass through sigmoid for probability-like scores.
    3. Fallback to binary predictions with a warning (PR-AUC will be poor).
    """
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        classes = list(pipeline.classes_)
        if fraud_label in classes:
            idx = classes.index(fraud_label)
        else:
            idx = 1  # best guess; emit a warning
            warnings.warn(
                f"Fraud label {fraud_label!r} not found in model.classes_ "
                f"{classes}. Using column index 1 as a fallback."
            )
        return proba[:, idx]

    if hasattr(pipeline, "decision_function"):
        raw = pipeline.decision_function(X)
        if raw.ndim == 2:  # multi-class OvR — take the fraud column
            classes = list(pipeline.classes_)
            idx = classes.index(fraud_label) if fraud_label in classes else 1
            raw = raw[:, idx]
        return _sigmoid(raw)

    warnings.warn(
        "Model supports neither predict_proba nor decision_function. "
        "Binary predictions will be used; PR-AUC may be misleading."
    )
    pred = pipeline.predict(X)
    return pred.astype(float)


def _choose_threshold(y_true, scores):
    """Return the threshold that maximises F1-score on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_idx = int(np.nanargmax(f1_values))
    return float(np.clip(thresholds[best_idx], 0.0, 1.0))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Place creditcard.csv in {DATA_PATH.parent}\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    REPORTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")
    if df.empty:
        raise ValueError("Dataset is empty.")

    X = df.drop(columns=["Class"]).replace([np.inf, -np.inf], np.nan)
    y = df["Class"].astype(int)

    # ---------------------------------------------------------------
    # Stratified split: 70 % train | 15 % validation | 15 % test.
    # The test set is created here but is NOT used in this script;
    # it is reserved for a final evaluation run only.
    # ---------------------------------------------------------------
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full,
        test_size=0.50, random_state=RANDOM_STATE, stratify=y_train_full
    )

    # Limit expensive algorithms to a reproducible subset of the training data.
    n_fit = min(60_000, len(X_train))
    X_fit, _, y_fit, _ = train_test_split(
        X_train, y_train,
        train_size=n_fit, random_state=RANDOM_STATE, stratify=y_train
    )

    models = {
        "Logistic Regression": create_logistic_regression_model(random_state=RANDOM_STATE),
        "KNN": create_knn_model(),
        "Naive Bayes": create_naive_bayes_model(),
        "SLP (Perceptron)": create_perceptron_model(random_state=RANDOM_STATE),
        "MLP": create_mlp_model(random_state=RANDOM_STATE),
        "Random Forest (extra)": create_random_forest_model(
            random_state=RANDOM_STATE, min_samples_leaf=1
        ),
    }

    rows = []
    fitted = {}

    for name, pipeline in models.items():
        print(f"\nTraining {name} ...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_fit, y_fit)
        except Exception as exc:
            raise RuntimeError(f"Failed to train model '{name}': {exc}") from exc

        # ---------------------------------------------------------------
        # Validation-set evaluation only – no test data here.
        # ---------------------------------------------------------------
        valid_pred = pipeline.predict(X_valid)
        valid_scores = _get_scores(pipeline, X_valid, fraud_label=1)

        threshold = _choose_threshold(y_valid.to_numpy(), valid_scores)
        threshold_pred = (valid_scores >= threshold).astype(int)

        try:
            val_pr_auc = float(average_precision_score(y_valid, valid_scores))
        except ValueError:
            val_pr_auc = float("nan")

        rows.append({
            "model": name,
            "validation_accuracy": accuracy_score(y_valid, valid_pred),
            "validation_precision": precision_score(
                y_valid, threshold_pred, zero_division=0
            ),
            "validation_recall": recall_score(
                y_valid, threshold_pred, zero_division=0
            ),
            "validation_f1": f1_score(
                y_valid, threshold_pred, zero_division=0
            ),
            "validation_pr_auc": val_pr_auc,
            "validation_threshold": threshold,
        })
        fitted[name] = pipeline
        print(classification_report(y_valid, threshold_pred, zero_division=0))

    results = pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)
    results.to_csv(REPORTS / "classification_comparison.csv", index=False)

    print("\n===== MODEL COMPARISON — validation set, sorted by PR-AUC =====")
    print(results.to_string(index=False))

    best_name = str(results.iloc[0]["model"])
    best_pipeline = fitted[best_name]
    best_threshold = float(results.iloc[0]["validation_threshold"])

    # Derive fraud class index safely.
    fraud_class_index = list(best_pipeline.classes_).index(1)

    # Save comparison-only model (does NOT overwrite the dashboard model).
    joblib.dump(
        {
            "model": best_pipeline,
            "model_name": best_name,
            "feature_columns": list(X.columns),
            "threshold": best_threshold,
            "fraud_class_index": fraud_class_index,
            "note": (
                "Comparison-only artifact. Dashboard model is produced by "
                "python -m src.train"
            ),
        },
        ARTIFACTS / "comparison_model.joblib",
    )

    with open(REPORTS / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"selected_model": best_name, "metrics": rows},
            f,
            indent=2,
        )

    # Confusion matrix on the validation set (test data stays untouched).
    cm = confusion_matrix(
        y_valid, (results.iloc[0]["validation_threshold"]
                  <= _get_scores(best_pipeline, X_valid)).astype(int),
        labels=[0, 1],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix (validation): {best_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], labels=["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], labels=["Legitimate", "Fraud"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png")
    plt.close(fig)

    print(f"\nSaved best comparison model: {best_name}")
    print(f"Comparison-only artifact: {ARTIFACTS / 'comparison_model.joblib'}")
    print("Dashboard model is only produced by: python -m src.train")


if __name__ == "__main__":
    main()
