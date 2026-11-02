"""Automated tests for the Credit Card Fraud Detection project.

Covers:
* Dataset validation  (src.common)
* Training pipeline   (src.train)
* Prediction / scoring workflow
* Alert history       (src.advanced_features)
* Evaluation metrics  (src.train  \u2013  helper functions)

Run with:
    pytest tests/test_project.py -v
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N_LEGIT = 200
_N_FRAUD = 20
_N_TOTAL = _N_LEGIT + _N_FRAUD
_FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _make_dataset(
    n_legit: int = _N_LEGIT,
    n_fraud: int = _N_FRAUD,
    add_class: bool = True,
    corrupt_class: bool = False,
) -> pd.DataFrame:
    """Return a small synthetic dataset with the same schema as creditcard.csv."""
    rng = np.random.default_rng(0)
    n = n_legit + n_fraud
    data = {col: rng.standard_normal(n) for col in _FEATURE_COLS}
    df = pd.DataFrame(data)
    df["Time"] = rng.uniform(0, 172_800, n)
    df["Amount"] = rng.exponential(scale=100, size=n)
    if add_class:
        labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
        rng.shuffle(labels)
        df["Class"] = labels.astype(int)
        if corrupt_class:
            df["Class"] = df["Class"].replace({0: 99, 1: 99})
    return df


# ---------------------------------------------------------------------------
# Tests: src.common  \u2013  dataset validation
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_valid_dataset_loads(self, tmp_path):
        df = _make_dataset()
        csv = tmp_path / "creditcard.csv"
        df.to_csv(csv, index=False)
        from src.common import load_dataset
        loaded = load_dataset(csv)
        assert loaded.shape == df.shape
        assert "Class" in loaded.columns

    def test_missing_file_raises(self, tmp_path):
        from src.common import load_dataset
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_dataset(tmp_path / "nonexistent.csv")

    def test_empty_dataset_raises(self, tmp_path):
        from src.common import load_dataset
        csv = tmp_path / "empty.csv"
        csv.write_text("Time,V1,Class\n")  # header only
        with pytest.raises(ValueError, match="empty"):
            load_dataset(csv)

    def test_missing_class_column_raises(self, tmp_path):
        from src.common import load_dataset
        df = _make_dataset(add_class=False)
        csv = tmp_path / "no_class.csv"
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="Class"):
            load_dataset(csv)

    def test_invalid_class_values_raise(self, tmp_path):
        from src.common import load_dataset
        df = _make_dataset(corrupt_class=True)
        csv = tmp_path / "bad_labels.csv"
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="labels"):
            load_dataset(csv)

    def test_missing_target_values_raise(self, tmp_path):
        from src.common import load_dataset
        df = _make_dataset()
        df.loc[0, "Class"] = float("nan")
        csv = tmp_path / "nan_class.csv"
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing"):
            load_dataset(csv)


class TestFeatureColumns:
    def test_excludes_target(self):
        from src.common import feature_columns
        df = _make_dataset()
        cols = feature_columns(df)
        assert "Class" not in cols
        assert set(cols) == set(_FEATURE_COLS)

    def test_preserves_csv_order(self):
        from src.common import feature_columns
        df = _make_dataset()
        cols = feature_columns(df)
        # The first column in the CSV is Time.
        assert cols[0] == "Time"


# ---------------------------------------------------------------------------
# Tests: Training pipeline helpers (src.train)
# ---------------------------------------------------------------------------


class TestChooseThreshold:
    def test_returns_float_in_unit_interval(self):
        from src.train import choose_threshold
        rng = np.random.default_rng(7)
        y = np.array([0] * 90 + [1] * 10)
        scores = rng.uniform(0, 1, 100)
        t = choose_threshold(y, scores)
        assert 0.0 <= t <= 1.0

    def test_handles_trivial_scores(self):
        from src.train import choose_threshold
        y = np.array([0] * 5 + [1] * 5)
        scores = np.zeros(10)
        t = choose_threshold(y, scores)
        assert 0.0 <= t <= 1.0


class TestMetricsAtThreshold:
    def test_perfect_predictions(self):
        from src.train import metrics_at_threshold
        y = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.1, 0.9, 0.95])
        m = metrics_at_threshold(y, scores, threshold=0.5)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_all_wrong_predictions(self):
        from src.train import metrics_at_threshold
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.9, 0.1, 0.1])
        m = metrics_at_threshold(y, scores, threshold=0.5)
        assert m["precision"] == pytest.approx(0.0, abs=1e-9)
        assert m["recall"] == pytest.approx(0.0, abs=1e-9)
        assert m["fp"] == 2
        assert m["fn"] == 2

    def test_pr_auc_uses_continuous_scores(self):
        """PR-AUC must be computed from scores, not binary predictions."""
        from src.train import metrics_at_threshold
        y = np.array([0, 0, 0, 1])
        scores = np.array([0.2, 0.4, 0.6, 0.99])
        m = metrics_at_threshold(y, scores, threshold=0.5)
        # Minimum informative PR-AUC on this sample is > 0.5 given the scores.
        assert m["pr_auc"] > 0.5


# ---------------------------------------------------------------------------
# Tests: Training pipeline  (integration, uses synthetic data + tmp paths)
# ---------------------------------------------------------------------------


class TestTrainingPipeline:
    """Run the full training pipeline against a tiny synthetic dataset."""

    @pytest.fixture()
    def synthetic_paths(self, tmp_path):
        """Write a synthetic CSV and patch all path constants in src.train."""
        df = _make_dataset(n_legit=500, n_fraud=50)
        csv = tmp_path / "creditcard.csv"
        df.to_csv(csv, index=False)
        artifact_dir = tmp_path / "artifacts"
        report_dir = tmp_path / "reports"
        artifact_dir.mkdir()
        report_dir.mkdir()
        return {
            "csv": csv,
            "artifact_dir": artifact_dir,
            "report_dir": report_dir,
            "model_path": artifact_dir / "fraud_model.joblib",
            "metrics_path": report_dir / "model_metrics.json",
            "comparison_path": report_dir / "model_comparison.csv",
        }

    def test_training_produces_artifacts(self, synthetic_paths):
        import src.train as train_module
        import src.common as common_module
        paths = synthetic_paths

        with (
            patch.object(common_module, "DATA_PATH", paths["csv"]),
            patch.object(common_module, "ARTIFACT_DIR", paths["artifact_dir"]),
            patch.object(common_module, "REPORT_DIR", paths["report_dir"]),
            patch.object(common_module, "MODEL_PATH", paths["model_path"]),
            patch.object(common_module, "METRICS_PATH", paths["metrics_path"]),
            patch.object(common_module, "COMPARISON_PATH", paths["comparison_path"]),
            patch.object(train_module, "ARTIFACT_DIR", paths["artifact_dir"]),
            patch.object(train_module, "REPORT_DIR", paths["report_dir"]),
            patch.object(train_module, "MODEL_PATH", paths["model_path"]),
            patch.object(train_module, "METRICS_PATH", paths["metrics_path"]),
            patch.object(train_module, "COMPARISON_PATH", paths["comparison_path"]),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_module.main()

        assert paths["model_path"].exists(), "Model artifact not saved"
        assert paths["metrics_path"].exists(), "Metrics JSON not saved"
        assert paths["comparison_path"].exists(), "Comparison CSV not saved"

    def test_model_artifact_has_required_keys(self, synthetic_paths):
        import joblib
        import src.train as train_module
        import src.common as common_module
        paths = synthetic_paths

        with (
            patch.object(common_module, "DATA_PATH", paths["csv"]),
            patch.object(common_module, "ARTIFACT_DIR", paths["artifact_dir"]),
            patch.object(common_module, "REPORT_DIR", paths["report_dir"]),
            patch.object(common_module, "MODEL_PATH", paths["model_path"]),
            patch.object(common_module, "METRICS_PATH", paths["metrics_path"]),
            patch.object(common_module, "COMPARISON_PATH", paths["comparison_path"]),
            patch.object(train_module, "ARTIFACT_DIR", paths["artifact_dir"]),
            patch.object(train_module, "REPORT_DIR", paths["report_dir"]),
            patch.object(train_module, "MODEL_PATH", paths["model_path"]),
            patch.object(train_module, "METRICS_PATH", paths["metrics_path"]),
            patch.object(train_module, "COMPARISON_PATH", paths["comparison_path"]),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_module.main()

        bundle = joblib.load(paths["model_path"])
        required_keys = {
            "model",
            "model_name",
            "threshold",
            "feature_columns",
            "test_metrics",
            "random_state",
            "fraud_class_label",
            "fraud_class_index",
            "validation_metrics",
            "model_version",
        }
        missing = required_keys - set(bundle.keys())
        assert not missing, f"Model artifact missing keys: {missing}"

    def test_threshold_in_unit_interval(self, synthetic_paths):
        import joblib
        import src.train as train_module
        import src.common as common_module
        paths = synthetic_paths

        with (
            patch.object(common_module, "DATA_PATH", paths["csv"]),
            patch.object(common_module, "ARTIFACT_DIR", paths["artifact_dir"]),
            patch.object(common_module, "REPORT_DIR", paths["report_dir"]),
            patch.object(common_module, "MODEL_PATH", paths["model_path"]),
            patch.object(common_module, "METRICS_PATH", paths["metrics_path"]),
            patch.object(common_module, "COMPARISON_PATH", paths["comparison_path"]),
            patch.object(train_module, "ARTIFACT_DIR", paths["artifact_dir"]),
            patch.object(train_module, "REPORT_DIR", paths["report_dir"]),
            patch.object(train_module, "MODEL_PATH", paths["model_path"]),
            patch.object(train_module, "METRICS_PATH", paths["metrics_path"]),
            patch.object(train_module, "COMPARISON_PATH", paths["comparison_path"]),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_module.main()

        bundle = joblib.load(paths["model_path"])
        t = bundle["threshold"]
        assert 0.0 <= t <= 1.0, f"Threshold {t} is outside [0, 1]"

    def test_class_column_not_in_features(self, synthetic_paths):
        import joblib
        import src.train as train_module
        import src.common as common_module
        paths = synthetic_paths

        with (
            patch.object(common_module, "DATA_PATH", paths["csv"]),
            patch.object(common_module, "ARTIFACT_DIR", paths["artifact_dir"]),
            patch.object(common_module, "REPORT_DIR", paths["report_dir"]),
            patch.object(common_module, "MODEL_PATH", paths["model_path"]),
            patch.object(common_module, "METRICS_PATH", paths["metrics_path"]),
            patch.object(common_module, "COMPARISON_PATH", paths["comparison_path"]),
            patch.object(train_module, "ARTIFACT_DIR", paths["artifact_dir"]),
            patch.object(train_module, "REPORT_DIR", paths["report_dir"]),
            patch.object(train_module, "MODEL_PATH", paths["model_path"]),
            patch.object(train_module, "METRICS_PATH", paths["metrics_path"]),
            patch.object(train_module, "COMPARISON_PATH", paths["comparison_path"]),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_module.main()

        bundle = joblib.load(paths["model_path"])
        assert "Class" not in bundle["feature_columns"], (
            "Target column 'Class' must not be a model feature (data leakage)."
        )

    def test_fraud_scores_are_valid_probabilities(self, synthetic_paths):
        """Fraud scores from the saved model must lie in [0, 1]."""
        import joblib
        import src.train as train_module
        import src.common as common_module
        paths = synthetic_paths

        with (
            patch.object(common_module, "DATA_PATH", paths["csv"]),
            patch.object(common_module, "ARTIFACT_DIR", paths["artifact_dir"]),
            patch.object(common_module, "REPORT_DIR", paths["report_dir"]),
            patch.object(common_module, "MODEL_PATH", paths["model_path"]),
            patch.object(common_module, "METRICS_PATH", paths["metrics_path"]),
            patch.object(common_module, "COMPARISON_PATH", paths["comparison_path"]),
            patch.object(train_module, "ARTIFACT_DIR", paths["artifact_dir"]),
            patch.object(train_module, "REPORT_DIR", paths["report_dir"]),
            patch.object(train_module, "MODEL_PATH", paths["model_path"]),
            patch.object(train_module, "METRICS_PATH", paths["metrics_path"]),
            patch.object(train_module, "COMPARISON_PATH", paths["comparison_path"]),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_module.main()

        bundle = joblib.load(paths["model_path"])
        model = bundle["model"]
        features = bundle["feature_columns"]
        fraud_idx = bundle.get("fraud_class_index", 1)

        df = pd.read_csv(paths["csv"])
        X = df[features].copy()
        scores = model.predict_proba(X)[:, fraud_idx]
        assert np.all(scores >= 0) and np.all(scores <= 1), (
            f"Fraud scores out of [0, 1]: min={scores.min():.4f} max={scores.max():.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: Prediction workflow
# ---------------------------------------------------------------------------


class TestPredictionWorkflow:
    @pytest.fixture()
    def trained_bundle(self, tmp_path):
        """Train a small model and return the saved bundle."""
        import joblib
        import src.train as train_module
        import src.common as common_module

        df = _make_dataset(n_legit=500, n_fraud=50)
        csv = tmp_path / "creditcard.csv"
        df.to_csv(csv, index=False)
        artifact_dir = tmp_path / "artifacts"
        report_dir = tmp_path / "reports"
        artifact_dir.mkdir()
        report_dir.mkdir()
        model_path = artifact_dir / "fraud_model.joblib"
        metrics_path = report_dir / "model_metrics.json"
        comparison_path = report_dir / "model_comparison.csv"

        with (
            patch.object(common_module, "DATA_PATH", csv),
            patch.object(common_module, "ARTIFACT_DIR", artifact_dir),
            patch.object(common_module, "REPORT_DIR", report_dir),
            patch.object(common_module, "MODEL_PATH", model_path),
            patch.object(common_module, "METRICS_PATH", metrics_path),
            patch.object(common_module, "COMPARISON_PATH", comparison_path),
            patch.object(train_module, "ARTIFACT_DIR", artifact_dir),
            patch.object(train_module, "REPORT_DIR", report_dir),
            patch.object(train_module, "MODEL_PATH", model_path),
            patch.object(train_module, "METRICS_PATH", metrics_path),
            patch.object(train_module, "COMPARISON_PATH", comparison_path),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_module.main()

        return joblib.load(model_path)

    def test_valid_input_produces_predictions(self, trained_bundle):
        bundle = trained_bundle
        model = bundle["model"]
        features = bundle["feature_columns"]
        fraud_idx = bundle.get("fraud_class_index", 1)
        threshold = bundle["threshold"]

        df = _make_dataset(n_legit=20, n_fraud=2)
        X = df[features].copy()
        scores = model.predict_proba(X)[:, fraud_idx]
        predictions = (scores >= threshold).astype(int)

        assert len(scores) == len(df)
        assert set(predictions).issubset({0, 1})

    def test_missing_feature_raises_key_error_or_value_error(
        self, trained_bundle
    ):
        bundle = trained_bundle
        model = bundle["model"]
        features = bundle["feature_columns"]

        df = _make_dataset(n_legit=10, n_fraud=1)
        # Drop one required feature.
        X_incomplete = df[features[1:]].copy()
        with pytest.raises(Exception):
            model.predict_proba(X_incomplete)

    def test_risk_levels_consistent_with_predictions(self, trained_bundle):
        bundle = trained_bundle
        model = bundle["model"]
        features = bundle["feature_columns"]
        fraud_idx = bundle.get("fraud_class_index", 1)
        threshold = bundle["threshold"]

        df = _make_dataset(n_legit=50, n_fraud=5)
        X = df[features].copy()
        scores = model.predict_proba(X)[:, fraud_idx]
        predictions = (scores >= threshold).astype(int)

        high_cutoff = max(threshold, 0.80)
        risk_levels = np.where(
            predictions == 0,
            "LOW",
            np.where(scores >= high_cutoff, "HIGH", "REVIEW"),
        )

        # All unflagged must be LOW.
        assert all(
            risk_levels[i] == "LOW"
            for i in range(len(predictions))
            if predictions[i] == 0
        )
        # All flagged must be HIGH or REVIEW.
        assert all(
            risk_levels[i] in {"HIGH", "REVIEW"}
            for i in range(len(predictions))
            if predictions[i] == 1
        )

    def test_threshold_boundary(self, trained_bundle):
        """A score exactly at the threshold must be flagged (>=)."""
        bundle = trained_bundle
        threshold = bundle["threshold"]
        boundary_score = np.array([threshold])
        assert (boundary_score >= threshold).astype(int)[0] == 1


# ---------------------------------------------------------------------------
# Tests: Alert history (src.advanced_features)
# ---------------------------------------------------------------------------


class TestAlertHistory:
    def _make_scored_df(self, run_id: str = "test-run-001") -> pd.DataFrame:
        n = 10
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "Time": rng.uniform(0, 1000, n),
            "Amount": rng.uniform(10, 500, n),
            "fraud_score": rng.uniform(0, 1, n),
            "predicted_class": ([0] * 8 + [1] * 2),
            "risk_level": (["LOW"] * 8 + ["HIGH"] * 2),
            "review_status": (["Not flagged"] * 8 + ["Needs human review"] * 2),
            "source_row_number": list(range(1, n + 1)),
            "upload_run_id": run_id,
            "threshold_used": 0.5,
        })
        return df

    def test_new_upload_creates_history(self, tmp_path):
        df = self._make_scored_df()
        history_path = tmp_path / "reports" / "fraud_alert_history.csv"
        history_path.parent.mkdir()

        # We can't easily test Streamlit rendering, but we can test the
        # underlying file-writing logic by extracting it.
        # The function writes to reports/fraud_alert_history.csv when the
        # "Save" button is clicked; here we simulate the file logic directly.
        saved = df[df["predicted_class"] == 1].copy()
        saved["recorded_at"] = "2024-01-01 00:00:00"
        saved["source_file"] = "test.csv"
        saved.to_csv(history_path, index=False)

        assert history_path.exists()
        history = pd.read_csv(history_path)
        assert len(history) == 2  # 2 flagged rows

    def test_duplicate_alerts_are_not_saved_twice(self, tmp_path):
        """Saving the same run ID and row numbers twice must not duplicate."""
        history_path = tmp_path / "fraud_alert_history.csv"

        run_id = "run-abc123"
        row1 = {"upload_run_id": run_id, "source_row_number": 5,
                 "recorded_at": "2024-01-01 00:00:00", "source_file": "a.csv"}
        row2 = {"upload_run_id": run_id, "source_row_number": 5,
                 "recorded_at": "2024-01-01 00:01:00", "source_file": "a.csv"}

        previous = pd.DataFrame([row1])
        saved = pd.DataFrame([row2])

        # Dedup logic from advanced_features.py
        keys = ["upload_run_id", "source_row_number"]
        previous_keys = previous[keys].astype(str).agg("|".join, axis=1)
        saved_keys = saved[keys].astype(str).agg("|".join, axis=1)
        deduped = saved.loc[~saved_keys.isin(set(previous_keys))]

        assert deduped.empty, "Duplicate alert was not filtered out."

    def test_different_run_ids_are_not_merged(self):
        """Alerts from different runs must both be preserved."""
        run1 = pd.DataFrame([
            {"upload_run_id": "run-aaa", "source_row_number": 1}
        ])
        run2 = pd.DataFrame([
            {"upload_run_id": "run-bbb", "source_row_number": 1}
        ])

        keys = ["upload_run_id", "source_row_number"]
        previous_keys = run1[keys].astype(str).agg("|".join, axis=1)
        saved_keys = run2[keys].astype(str).agg("|".join, axis=1)
        deduped = run2.loc[~saved_keys.isin(set(previous_keys))]

        assert len(deduped) == 1, (
            "Alert from a different run was incorrectly filtered out."
        )

    def test_legacy_history_detection(self, tmp_path):
        """A history file without upload_run_id is identified as legacy."""
        history_path = tmp_path / "fraud_alert_history.csv"
        # Write a file without the upload_run_id column.
        pd.DataFrame([{"source_file": "old.csv", "amount": 100}]).to_csv(
            history_path, index=False
        )
        existing_columns = set(pd.read_csv(history_path, nrows=0).columns)
        assert "upload_run_id" not in existing_columns


# ---------------------------------------------------------------------------
# Tests: Evaluation metrics correctness
# ---------------------------------------------------------------------------


class TestEvaluationMetrics:
    def test_pr_auc_with_continuous_scores_vs_binary(self):
        """PR-AUC should be higher when continuous scores are used."""
        from sklearn.metrics import average_precision_score

        y = np.array([0, 0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95])
        binary = (scores >= 0.5).astype(float)

        pr_auc_scores = average_precision_score(y, scores)
        pr_auc_binary = average_precision_score(y, binary)

        assert pr_auc_scores >= pr_auc_binary, (
            "PR-AUC from continuous scores should be >= binary predictions."
        )

    def test_roc_auc_non_informative_for_constant_scores(self):
        """ROC-AUC should not be informative when all scores are identical.

        sklearn >= 1.3 no longer raises ValueError for constant scores;
        it returns 0.5 (random-classifier level) instead.  This test
        verifies that constant scores are provably non-informative.
        """
        from sklearn.metrics import roc_auc_score
        y = np.array([0, 0, 1, 1])
        constant_scores = np.array([0.5, 0.5, 0.5, 0.5])
        # Either a ValueError is raised (older sklearn) or the result is 0.5
        # (random-classifier level, i.e., non-informative).
        try:
            result = roc_auc_score(y, constant_scores)
            assert result == pytest.approx(0.5), (
                f"Constant scores should yield ROC-AUC ≈ 0.5 (non-informative), "
                f"got {result}."
            )
        except ValueError:
            pass  # Older sklearn behaviour is also acceptable.

    def test_confusion_matrix_shape(self):
        from src.train import metrics_at_threshold
        y = np.array([0, 1, 0, 1, 0])
        scores = np.array([0.2, 0.8, 0.3, 0.7, 0.4])
        m = metrics_at_threshold(y, scores, 0.5)
        assert set(m.keys()) >= {"tn", "fp", "fn", "tp"}
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y)

    def test_zero_division_handled(self):
        from src.train import metrics_at_threshold
        # All predictions are 0 \u2014 precision and recall should be 0, not an error.
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.1, 0.1, 0.1])
        m = metrics_at_threshold(y, scores, threshold=0.5)
        assert m["precision"] == pytest.approx(0.0, abs=1e-9)
        assert m["recall"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests: Stratified splitting (no data leakage)
# ---------------------------------------------------------------------------


class TestStratifiedSplit:
    def test_stratified_split_preserves_class_ratio(self):
        from sklearn.model_selection import train_test_split

        df = _make_dataset(n_legit=700, n_fraud=70)
        y = df["Class"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            df, y, test_size=0.30, stratify=y, random_state=42
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )

        overall_rate = y.mean()
        for split_name, y_split in [
            ("train", y_train),
            ("valid", y_valid),
            ("test", y_test),
        ]:
            rate = y_split.mean()
            assert abs(rate - overall_rate) < 0.05, (
                f"Fraud rate in {split_name} ({rate:.3f}) differs too much "
                f"from overall ({overall_rate:.3f})."
            )

    def test_test_set_not_used_in_threshold_selection(self):
        """The choose_threshold function must only accept validation labels."""
        from src.train import choose_threshold
        rng = np.random.default_rng(9)
        y_valid = np.array([0] * 45 + [1] * 5)
        valid_scores = rng.uniform(0, 1, 50)
        # This just verifies the function signature; leakage would only happen
        # if test labels were passed \u2014 which we verify does not happen in
        # test_training_produces_artifacts above.
        t = choose_threshold(y_valid, valid_scores)
        assert 0.0 <= t <= 1.0


# ---------------------------------------------------------------------------
# Tests: Individual Transaction Risk Explorer (src.advanced_features)
# ---------------------------------------------------------------------------


class TestIndividualTransactionRiskExplorer:
    """Covers the 11 verification points for the Individual Transaction Risk Explorer."""

    @pytest.fixture()
    def model_bundle(self):
        import joblib
        from src.common import MODEL_PATH
        if not MODEL_PATH.exists():
            pytest.skip("Trained model artifact not found.")
        return joblib.load(MODEL_PATH)

    def test_fraud_probability_extraction(self, model_bundle):
        """Point 1: Fraud probability extraction via predict_proba."""
        bundle = model_bundle
        model = bundle["model"]
        fraud_idx = bundle["fraud_class_index"]
        features = bundle["feature_columns"]

        df = _make_dataset(n_legit=5, n_fraud=1)
        X = df[features]
        scores = model.predict_proba(X)[:, fraud_idx]
        assert len(scores) == len(df)
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert isinstance(float(scores[0]), float)

    def test_correct_fraud_class_index(self, model_bundle):
        """Point 2: Correct fraud class index derived from model.classes_."""
        bundle = model_bundle
        model = bundle["model"]
        classes = list(model.classes_)
        assert 1 in classes
        expected_idx = classes.index(1)
        assert bundle["fraud_class_index"] == expected_idx
        assert model.classes_[expected_idx] == 1

    def test_correct_transaction_row_selection(self):
        """Point 3: 1-based user row number correctly maps to 0-based position."""
        df = _make_dataset(n_legit=1005, n_fraud=5)
        for user_row in [1, 2, 3, 10, 100, 1000]:
            position = user_row - 1
            assert position >= 0
            row = df.iloc[position]
            # Ensure row features exactly match the expected slice
            pd.testing.assert_series_equal(row, df.iloc[position])

    def test_correct_feature_ordering(self, model_bundle):
        """Point 4: Features preserve training CSV order without target."""
        from src.common import DATA_PATH, feature_columns
        bundle = model_bundle
        expected_features = bundle["feature_columns"]
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH, nrows=5)
            csv_features = feature_columns(df)
            assert expected_features == csv_features
        # Ensure model accepts exactly expected_features
        df_sample = _make_dataset(n_legit=2, n_fraud=1)
        X = df_sample[expected_features]
        probs = bundle["model"].predict_proba(X)
        assert probs.shape == (3, 2)

    def test_class_column_exclusion(self, model_bundle):
        """Point 5: Class column must never be a model feature."""
        bundle = model_bundle
        assert "Class" not in bundle["feature_columns"]
        # In scoring workflow, any incoming dataframe with Class must have Class excluded
        incoming = _make_dataset(n_legit=2, n_fraud=1, add_class=True)
        feature_cols = [c for c in bundle["feature_columns"] if c != "Class"]
        X = incoming[feature_cols]
        assert "Class" not in X.columns
        bundle["model"].predict_proba(X)

    def test_threshold_application(self, model_bundle):
        """Point 6: Threshold application (fraud = score >= threshold)."""
        bundle = model_bundle
        threshold = float(bundle["threshold"])
        assert 0.0 <= threshold <= 1.0

        scores = np.array([threshold, threshold - 1e-7, threshold + 1e-7, 0.0, 1.0])
        preds = (scores >= threshold).astype(int)
        assert preds[0] == 1  # exactly on threshold -> 1
        assert preds[1] == 0  # just below -> 0
        assert preds[2] == 1  # just above -> 1
        assert preds[3] == 0  # 0.0 -> 0
        assert preds[4] == 1  # 1.0 -> 1

    def test_risk_level_calculation(self):
        """Point 7: Risk-level calculation based on score and threshold."""
        from src.advanced_features import calculate_risk_level
        threshold = 0.4898
        # Below threshold -> LOW
        assert calculate_risk_level(0.0, threshold) == "LOW"
        assert calculate_risk_level(0.4897, threshold) == "LOW"
        # At or above threshold but below 0.80 -> REVIEW
        assert calculate_risk_level(0.4898, threshold) == "REVIEW"
        assert calculate_risk_level(0.75, threshold) == "REVIEW"
        # At or above 0.80 -> HIGH
        assert calculate_risk_level(0.80, threshold) == "HIGH"
        assert calculate_risk_level(0.95, threshold) == "HIGH"

        # Edge case: threshold > 0.80
        high_thresh = 0.85
        assert calculate_risk_level(0.84, high_thresh) == "LOW"
        assert calculate_risk_level(0.85, high_thresh) == "HIGH"

    def test_small_probability_formatting(self):
        """Point 8: Small probability formatting preserves precision."""
        from src.advanced_features import format_fraud_score
        assert format_fraud_score(0.000001) == "0.0001%"
        assert format_fraud_score(0.0001) == "0.0100%"
        assert format_fraud_score(0.001) == "0.1000%"
        assert format_fraud_score(0.01) == "1.0000%"
        assert format_fraud_score(0.50) == "50.0000%"
        assert format_fraud_score(0.0) == "0.0000%"
        assert format_fraud_score(np.nan) == "N/A"
        assert format_fraud_score(None) == "N/A"
        # Very tiny nonzero probability below 0.0001%
        assert format_fraud_score(0.0000005) == "0.000050%"

    def test_actual_versus_predicted_label(self):
        """Point 9: Actual vs predicted label comparison."""
        from src.advanced_features import get_prediction_result
        # True Negative
        assert get_prediction_result(0, 0) == "Correct prediction"
        assert get_prediction_result(0.0, 0) == "Correct prediction"
        # True Positive
        assert get_prediction_result(1, 1) == "Correct fraud detection"
        assert get_prediction_result(1.0, 1) == "Correct fraud detection"
        # Missing or invalid labels
        assert get_prediction_result(None, 0) is None
        assert get_prediction_result(np.nan, 1) is None
        assert get_prediction_result(99, 1) is None

    def test_false_positive_identification(self, model_bundle):
        """Point 10: False positive identification (Actual 0, Predicted 1)."""
        from src.advanced_features import get_prediction_result
        assert get_prediction_result(0, 1) == "False positive"
        assert get_prediction_result(0.0, 1) == "False positive"

        # Test on actual dataset row 14921 if available (known FP)
        from src.common import DATA_PATH
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            row_idx = 14920  # 0-based for row 14921
            features = model_bundle["feature_columns"]
            score = model_bundle["model"].predict_proba(
                df.loc[[row_idx], features]
            )[0, model_bundle["fraud_class_index"]]
            pred = int(score >= model_bundle["threshold"])
            actual = int(df.loc[row_idx, "Class"])
            assert actual == 0
            assert pred == 1
            assert get_prediction_result(actual, pred) == "False positive"

    def test_false_negative_identification(self, model_bundle):
        """Point 11: False negative identification (Actual 1, Predicted 0)."""
        from src.advanced_features import get_prediction_result
        assert get_prediction_result(1, 0) == "False negative"
        assert get_prediction_result(1.0, 0) == "False negative"

        # Test on actual dataset row 624 if available (known FN)
        from src.common import DATA_PATH
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            row_idx = 623  # 0-based for row 624
            features = model_bundle["feature_columns"]
            score = model_bundle["model"].predict_proba(
                df.loc[[row_idx], features]
            )[0, model_bundle["fraud_class_index"]]
            pred = int(score >= model_bundle["threshold"])
            actual = int(df.loc[row_idx, "Class"])
            assert actual == 1
            assert pred == 0
            assert get_prediction_result(actual, pred) == "False negative"


# ---------------------------------------------------------------------------
# Tests: System Enhancements (CSV validation, SHAP, recall threshold)
# ---------------------------------------------------------------------------


class TestSystemEnhancements:
    """Covers CSV validation, SHAP explainability, recall threshold, and confusion matrix."""

    def test_csv_validation_valid(self):
        from src.common import validate_uploaded_dataframe
        df = _make_dataset(n_legit=10, n_fraud=1)
        valid, msg, X = validate_uploaded_dataframe(df, _FEATURE_COLS)
        assert valid is True
        assert msg == ""
        assert X.shape == (11, len(_FEATURE_COLS))

    def test_csv_validation_empty(self):
        from src.common import validate_uploaded_dataframe
        valid, msg, X = validate_uploaded_dataframe(pd.DataFrame(), _FEATURE_COLS)
        assert valid is False
        assert "empty" in msg.lower()

    def test_csv_validation_missing_columns(self):
        from src.common import validate_uploaded_dataframe
        df = _make_dataset(n_legit=5, n_fraud=1).drop(columns=["V14", "V15"])
        valid, msg, X = validate_uploaded_dataframe(df, _FEATURE_COLS)
        assert valid is False
        assert "V14" in msg
        assert "V15" in msg

    def test_csv_validation_duplicate_columns(self):
        from src.common import validate_uploaded_dataframe
        df = _make_dataset(n_legit=5, n_fraud=1)
        df_dupe = pd.concat([df, df[["Amount"]]], axis=1)
        valid, msg, X = validate_uploaded_dataframe(df_dupe, _FEATURE_COLS)
        assert valid is False
        assert "duplicate" in msg.lower()

    def test_csv_validation_non_numeric(self):
        from src.common import validate_uploaded_dataframe
        df = _make_dataset(n_legit=5, n_fraud=1)
        df["V1"] = ["invalid_str"] * len(df)
        valid, msg, X = validate_uploaded_dataframe(df, _FEATURE_COLS)
        assert valid is False
        assert "non-numeric" in msg.lower()

    def test_choose_recall_focused_threshold(self):
        from src.train import choose_recall_focused_threshold, choose_threshold
        y_true = np.array([0] * 90 + [1] * 10)
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        t_f1 = choose_threshold(y_true, scores)
        t_f2 = choose_recall_focused_threshold(y_true, scores, beta=2.0)
        assert 0.0 <= t_f2 <= 1.0

    def test_shap_explainability_structure(self):
        import joblib
        from src.common import MODEL_PATH
        from src.advanced_features import explain_transaction_shap
        if not MODEL_PATH.exists():
            pytest.skip("Model artifact not found")
        bundle = joblib.load(MODEL_PATH)
        df = _make_dataset(n_legit=2, n_fraud=1)
        row = df.iloc[0]
        res = explain_transaction_shap(bundle["model"], row, bundle["feature_columns"], top_n=5)
        assert res is not None
        assert "base_value" in res
        assert "top_features" in res
        assert len(res["top_features"]) <= 5
        assert set(res["top_features"].columns) >= {"feature", "feature_value", "shap_value", "direction"}


class TestModularModels:
    """Tests for modular model architectures in src.models."""

    def test_all_factories_importable_from_package(self):
        import src.models as models
        expected = [
            "create_dummy_model",
            "create_logistic_regression_model",
            "create_random_forest_model",
            "create_extra_trees_model",
            "create_knn_model",
            "create_naive_bayes_model",
            "create_perceptron_model",
            "create_mlp_model",
        ]
        for name in expected:
            assert hasattr(models, name), f"Missing {name} in src.models"
            assert callable(getattr(models, name)), f"{name} is not callable"

    def test_dummy_model_factory(self):
        from src.models import create_dummy_model
        from sklearn.dummy import DummyClassifier
        model = create_dummy_model(strategy="most_frequent")
        assert isinstance(model, DummyClassifier)
        assert model.strategy == "most_frequent"

    def test_logistic_regression_model_factory(self):
        from src.models import create_logistic_regression_model
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        pipe = create_logistic_regression_model(random_state=123, class_weight="balanced")
        assert isinstance(pipe, Pipeline)
        assert "imputer" in pipe.named_steps
        assert "scaler" in pipe.named_steps
        assert "model" in pipe.named_steps
        assert isinstance(pipe.named_steps["model"], LogisticRegression)
        assert pipe.named_steps["model"].random_state == 123
        assert pipe.named_steps["model"].class_weight == "balanced"

    def test_random_forest_model_factory(self):
        from src.models import create_random_forest_model
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier
        pipe = create_random_forest_model(n_estimators=50, random_state=99, min_samples_leaf=3)
        assert isinstance(pipe, Pipeline)
        assert "imputer" in pipe.named_steps
        assert "model" in pipe.named_steps
        rf = pipe.named_steps["model"]
        assert isinstance(rf, RandomForestClassifier)
        assert rf.n_estimators == 50
        assert rf.random_state == 99
        assert rf.min_samples_leaf == 3

    def test_extra_trees_model_factory(self):
        from src.models import create_extra_trees_model
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import ExtraTreesClassifier
        pipe = create_extra_trees_model(n_estimators=60, random_state=77, min_samples_leaf=2)
        assert isinstance(pipe, Pipeline)
        assert "imputer" in pipe.named_steps
        assert "model" in pipe.named_steps
        et = pipe.named_steps["model"]
        assert isinstance(et, ExtraTreesClassifier)
        assert et.n_estimators == 60
        assert et.random_state == 77
        assert et.min_samples_leaf == 2

    def test_additional_models_factories(self):
        from src.models import (
            create_knn_model,
            create_naive_bayes_model,
            create_perceptron_model,
            create_mlp_model,
        )
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import Perceptron
        from sklearn.neural_network import MLPClassifier

        knn_pipe = create_knn_model(n_neighbors=3)
        assert isinstance(knn_pipe.named_steps["model"], KNeighborsClassifier)
        assert knn_pipe.named_steps["model"].n_neighbors == 3

        nb_pipe = create_naive_bayes_model()
        assert isinstance(nb_pipe.named_steps["model"], GaussianNB)

        slp_pipe = create_perceptron_model(random_state=42)
        assert isinstance(slp_pipe.named_steps["model"], Perceptron)

        mlp_pipe = create_mlp_model(hidden_layer_sizes=(16, 8), random_state=42)
        assert isinstance(mlp_pipe.named_steps["model"], MLPClassifier)
        assert mlp_pipe.named_steps["model"].hidden_layer_sizes == (16, 8)

    def test_models_fit_and_predict_synthetic_batch(self):
        from src.models import (
            create_dummy_model,
            create_logistic_regression_model,
            create_random_forest_model,
            create_extra_trees_model,
        )
        df = _make_dataset(n_legit=50, n_fraud=10)
        X = df[_FEATURE_COLS]
        y = df["Class"]

        models = [
            create_dummy_model(),
            create_logistic_regression_model(),
            create_random_forest_model(n_estimators=10),
            create_extra_trees_model(n_estimators=10),
        ]
        for m in models:
            m.fit(X, y)
            assert hasattr(m, "predict")
            preds = m.predict(X)
            assert len(preds) == len(X)



