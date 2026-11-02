# Test Report — Credit Card Fraud Detection ML Engine

**Test run date:** 2026-10-28  
**Environment:** Python 3.12.10, pytest 9.2.0, scikit-learn 1.9.0, pandas 3.0.3, numpy 2.5.1  
**Command:** `python -m pytest tests/test_project.py -v --tb=short`  
**Dataset used:** Synthetic (generated in-memory; real `data/creditcard.csv` not required)

---

## Summary

| Category | Count |
|---|-----------|
| **Tests collected** | 32 |
| **Passed** | 32 |
| **Failed** | 0 |
| **Skipped** | 0 |
| **Errors** | 0 |
| **Duration** | ~4 minutes (5× integration tests on synthetic data + 23 fast unit tests) |

---

## Test Results by Class

### `TestLoadDataset` — Dataset Validation (6 tests)

| Test | Result | Notes |
|---|---|---|
| `test_valid_dataset_loads` | PASSED | Synthetic CSV with all 31 columns loads successfully |
| `test_missing_file_raises` | PASSED | `FileNotFoundError` raised with descriptive message |
| `test_empty_dataset_raises` | PASSED | `ValueError` raised for header-only CSV |
| `test_missing_class_column_raises` | PASSED | `ValueError` raised when Class column absent |
| `test_invalid_class_values_raise` | PASSED | `ValueError` raised when labels are not 0/1 |
| `test_missing_target_values_raise` | PASSED | `ValueError` raised when Class contains NaN |

### `TestFeatureColumns` — Feature Extraction (2 tests)

| Test | Result | Notes |
|---|---|---|
| `test_excludes_target` | PASSED | `Class` not in returned feature list |
| `test_preserves_csv_order` | PASSED | First column is `Time` as in CSV |

### `TestChooseThreshold` — Threshold Selection (2 tests)

| Test | Result | Notes |
|---|---|---|
| `test_returns_float_in_unit_interval` | PASSED | Threshold in [0.0, 1.0] for random scores |
| `test_handles_trivial_scores` | PASSED | All-zero scores handled without crash |

### `TestMetricsAtThreshold` — Metric Computation (3 tests)

| Test | Result | Notes |
|---|---|---|
| `test_perfect_predictions` | PASSED | Precision=1.0, Recall=1.0, F1=1.0, TP=2, FP=0, FN=0 |
| `test_all_wrong_predictions` | PASSED | Precision=0.0, Recall=0.0, FP=2, FN=2 |
| `test_pr_auc_uses_continuous_scores` | PASSED | PR-AUC > 0.5 confirmed for informative scores |

### `TestTrainingPipeline` — Integration Tests (5 tests)

These tests run the complete training pipeline on a 550-row synthetic dataset (500 legitimate + 50 fraud).

| Test | Result | Notes |
|---|---|---|
| `test_training_produces_artifacts` | PASSED | `fraud_model.joblib`, `model_metrics.json`, `model_comparison.csv` created |
| `test_model_artifact_has_required_keys` | PASSED | All 10 required keys present in artifact |
| `test_threshold_in_unit_interval` | PASSED | Saved threshold ∈ [0.0, 1.0] |
| `test_class_column_not_in_features` | PASSED | `Class` not in `feature_columns` (no target leakage) |
| `test_fraud_scores_are_valid_probabilities` | PASSED | All scores ∈ [0.0, 1.0] |

### `TestPredictionWorkflow` — Scoring Workflow (4 tests)

| Test | Result | Notes |
|---|---|---|
| `test_valid_input_produces_predictions` | PASSED | 22 rows scored; predictions are 0 or 1 |
| `test_missing_feature_raises_key_error_or_value_error` | PASSED | Exception raised when feature is missing |
| `test_risk_levels_consistent_with_predictions` | PASSED | LOW↔unflagged, HIGH/REVIEW↔flagged |
| `test_threshold_boundary` | PASSED | Score = threshold → predicted class = 1 |

### `TestAlertHistory` — Fraud Alert History (4 tests)

| Test | Result | Notes |
|---|---|---|
| `test_new_upload_creates_history` | PASSED | 2 flagged rows written to CSV |
| `test_duplicate_alerts_are_not_saved_twice` | PASSED | Same run_id + row_number filtered out |
| `test_different_run_ids_are_not_merged` | PASSED | Both alerts preserved when run IDs differ |
| `test_legacy_history_detection` | PASSED | Old file without `upload_run_id` detected |

### `TestEvaluationMetrics` — Metric Correctness (4 tests)

| Test | Result | Notes |
|---|---|---|
| `test_pr_auc_with_continuous_scores_vs_binary` | PASSED | PR-AUC(continuous) ≥ PR-AUC(binary) confirmed |
| `test_roc_auc_non_informative_for_constant_scores` | PASSED | sklearn 1.9+ returns 0.5 for constant scores (non-informative); ValueError accepted from older versions |
| `test_confusion_matrix_shape` | PASSED | tn+fp+fn+tp = n_samples |
| `test_zero_division_handled` | PASSED | No error when all predictions are negative |

### `TestStratifiedSplit` — Splitting Correctness (2 tests)

| Test | Result | Notes |
|---|---|---|
| `test_stratified_split_preserves_class_ratio` | PASSED | Fraud rate in each split within 5% of overall rate |
| `test_test_set_not_used_in_threshold_selection` | PASSED | `choose_threshold` accepts only validation labels |

---

## Runtime Issues

- **Long training integration tests**: Each of the 5 `TestTrainingPipeline` tests runs the full pipeline on synthetic data, which involves training 4 sklearn models. On a 550-row synthetic dataset this takes approximately 20–30 seconds per test. On the real 284,807-row dataset, `python -m src.train` takes approximately 2–5 minutes depending on CPU.
- **sklearn InconsistentVersionWarning**: The pre-committed `artifacts/fraud_model.joblib` was trained with sklearn 1.9.1 but the installed version is 1.9.0. This generates warnings (suppressed by `warnings.simplefilter("ignore")` during tests). Running `python -m src.train` regenerates the artifact in the current environment.

---

## Coverage

| Module | Covered | Notes |
|---|---|---|
| `src/common.py` | Load, validation, feature extraction | Full coverage |
| `src/train.py` | Full pipeline, metrics, threshold | Full coverage via integration |
| `src/classification.py` | Not directly tested | Covered by inspection; integration test would take 15+ minutes on synthetic data |
| `src/evaluation.py` | File discovery, fallback logic | Not directly tested (reads CSV files) |
| `src/advanced_features.py` | Alert history logic | Partial (Streamlit UI not testable without browser) |
| `src/clustering.py` | Not directly tested | Verified by inspection |
| `src/dashboard_extras.py` | Not directly tested | Streamlit rendering not testable without browser |
| `app.py` | Not directly tested | Streamlit app; requires browser automation |

---

## Remaining Limitations

1. **Streamlit UI tests**: The dashboard (`app.py`, `src/dashboard_extras.py`, `src/advanced_features.py` UI components) cannot be tested without Selenium or a Streamlit testing framework such as `streamlit-test`. These are not included due to the added complexity.
2. **`src/classification.py` integration test**: Running all six classifiers on even a small synthetic dataset takes significant time. A targeted unit test for the `_get_scores()` and `_choose_threshold()` helpers would be a good addition.
3. **Real dataset validation**: The test suite uses synthetic data for speed and portability. Results on the real 284,807-row dataset should be verified by running `python -m src.train` and inspecting `reports/model_metrics.json`.
4. **Concurrent alert history writes**: Multiple simultaneous Streamlit sessions writing to `fraud_alert_history.csv` could cause file corruption. File locking was not implemented; this is noted as a known limitation for a production deployment.
