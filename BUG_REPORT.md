# Bug Report — Credit Card Fraud Detection ML Engine

**Review date:** 2026-10-28  
**Reviewer:** Automated code audit  
**Scope:** All Python source files, requirements, and model artifacts

---

## BUG-001 — Data Leakage in `src/classification.py`: Test Set Used for Model Selection

| Field | Detail |
|---|---|
| **File** | `src/classification.py` |
| **Severity** | Critical |
| **Category** | ML Methodology / Data Leakage |

### Problem Description
Model comparison metrics (accuracy, precision, recall, F1, PR-AUC) were computed on `X_test` / `y_test`.  
The same test set was also used implicitly to rank models and choose the best one.

### Root Cause
A single `train_test_split` (80/20) was performed, and all per-model metrics were evaluated directly on the 20% test split. There was no separate validation set.

### Fix Implemented
Restructured to a proper 70/15/15 stratified train/validation/test split:
- Models are trained on the 70% training subset.
- Model comparison metrics are computed on the **15% validation set only**.
- The test set is created but **not used** in `classification.py` at all.

### Verification
Test `TestStratifiedSplit::test_stratified_split_preserves_class_ratio` passes.

---

## BUG-002 — Wrong PR-AUC for SLP/Perceptron in `src/classification.py`

| Field | Detail |
|---|---|
| **File** | `src/classification.py` |
| **Severity** | High |
| **Category** | Incorrect Metric Calculation |

### Problem Description
For models supporting neither `predict_proba` nor `decision_function`, the binary prediction array `pred` (integer 0/1) was used directly as the score for `average_precision_score`. PR-AUC from binary predictions is a point estimate and is always lower than PR-AUC from continuous scores.

### Root Cause
The fallback branch `else pred` did not convert predictions to continuous scores. Perceptron's `decision_function` is available through the Pipeline but the fallback still needed the sigmoid conversion.

### Fix Implemented
Introduced `_get_scores()` helper and `_sigmoid()` utility in `classification.py`:
1. Uses `predict_proba` if available, deriving fraud class index from `model.classes_`.
2. Uses `decision_function` via sigmoid conversion for Perceptron and similar.
3. Emits `warnings.warn` if forced to fall back to binary predictions.

### Verification
Test `TestEvaluationMetrics::test_pr_auc_with_continuous_scores_vs_binary` passes.

---

## BUG-003 — Hard-coded `predict_proba(X)[:, 1]` in `src/train.py`

| Field | Detail |
|---|---|
| **File** | `src/train.py` |
| **Severity** | Medium |
| **Category** | Incorrect Fraud Class Index |

### Problem Description
Column index 1 was hard-coded as the fraud class in both validation and test score extraction. If sklearn's internal label ordering differs, column 1 retrieves the wrong probability.

### Root Cause
`model.classes_` was not consulted when selecting the probability column.

### Fix Implemented
```python
classes = list(model.classes_)
fraud_idx = classes.index(1) if 1 in classes else 1
valid_scores = model.predict_proba(X_valid)[:, fraud_idx]
```
`fraud_class_index` is now stored in the model artifact.

### Verification
`test_fraud_scores_are_valid_probabilities` confirms scores are in [0, 1].

---

## BUG-004 — `AttributeError`: `DataFrame.get()` in `src/advanced_features.py`

| Field | Detail |
|---|---|
| **File** | `src/advanced_features.py` |
| **Severity** | High (runtime crash on "Save alerts" button) |
| **Category** | Python API Misuse |

### Problem Description
```python
saved["threshold_used"] = float(threshold) if threshold is not None \
                          else saved.get("threshold_used", np.nan)
```
`pd.DataFrame.get(col, default)` returns a Series (not a scalar). Assigning a misaligned Series to a column causes unexpected behaviour or errors. If `threshold` is `None` and the column doesn't exist, `DataFrame.get()` returns `None`, not `np.nan`.

### Root Cause
Confusion between `dict.get()` and `pd.DataFrame.get()` semantics.

### Fix Implemented
```python
if "threshold_used" not in saved.columns or saved["threshold_used"].isna().all():
    saved["threshold_used"] = float(threshold) if threshold is not None \
                              else float("nan")
```

### Verification
`TestAlertHistory` tests exercise the save path.

---

## BUG-005 — Wrong `source_row_number` Assignment in `src/advanced_features.py`

| Field | Detail |
|---|---|
| **File** | `src/advanced_features.py` |
| **Severity** | Medium |
| **Category** | Logic Error |

### Problem Description
```python
saved["source_row_number"] = [
    int(pos) + 1 for pos in range(len(df)) if predictions.iloc[pos] == 1
]
```
If `predictions` has NaN values, the list could have fewer entries than `len(saved)`, causing a `ValueError` on assignment.

### Root Cause
No length validation after the list comprehension.

### Fix Implemented
Added length check with a safe fallback using `saved.index + 1`.

### Verification
`TestAlertHistory::test_new_upload_creates_history` passes.

---

## BUG-006 — `src/evaluation.py` Reads Wrong CSV File

| Field | Detail |
|---|---|
| **File** | `src/evaluation.py` |
| **Severity** | Medium |
| **Category** | Wrong File Path |

### Problem Description
```python
path = REPORTS / "classification_comparison.csv"
```
`src.train` produces `model_comparison.csv`. Running `evaluation.py` after `src.train` without also running `src.classification` gave an unhelpful error message.

### Fix Implemented
Now tries `model_comparison.csv` first, falls back to `classification_comparison.csv`. Also reads and prints `model_metrics.json`.

### Verification
`test_training_produces_artifacts` verifies `model_comparison.csv` is created.

---

## BUG-007 — Missing `threshold` Key in Comparison Artifact

| Field | Detail |
|---|---|
| **File** | `src/classification.py` |
| **Severity** | Medium |
| **Category** | Incomplete Artifact |

### Problem Description
`comparison_model.joblib` lacked a `threshold` key. Dashboard code checking for `"threshold" not in bundle` would fail.

### Fix Implemented
Added `threshold`, `fraud_class_index`, and `note` to the comparison artifact.

---

## BUG-008 — Missing Metadata Keys in Dashboard Model Artifact

| Field | Detail |
|---|---|
| **File** | `src/train.py` |
| **Severity** | Low |
| **Category** | Incomplete Artifact Metadata |

### Problem Description
`fraud_model.joblib` was missing `validation_metrics`, `fraud_class_label`, `fraud_class_index`, and `model_version`.

### Fix Implemented
All four keys are now saved in the artifact.

### Verification
`test_model_artifact_has_required_keys` checks all required keys.

---

## BUG-009 — `src/clustering.py` Missing Value and Size Guards

| Field | Detail |
|---|---|
| **File** | `src/clustering.py` |
| **Severity** | Low |
| **Category** | Missing Input Validation |

### Problem Description
No handling for missing values, infinite values, datasets with fewer than 10 valid rows, or PCA failing on fewer than 2 numeric features.

### Fix Implemented
- `replace([inf, -inf], nan)` and `notna().all(axis=1)` filter before scaling
- `n_components = min(2, X_scaled.shape[1])` guard
- Minimum row count validation

---

## BUG-010 — `requirements.txt` Missing `pytest`

| Field | Detail |
|---|---|
| **File** | `requirements.txt` |
| **Severity** | Low |
| **Category** | Missing Dependency |

### Fix Implemented
Added `pytest>=7.0` and created comprehensive test suite at `tests/test_project.py`.

---

## BUG-011 — sklearn Version Mismatch Warning on Model Load

| Field | Detail |
|---|---|
| **File** | `artifacts/fraud_model.joblib` (pre-existing) |
| **Severity** | Low (warning, not crash) |
| **Category** | Version Incompatibility |

### Problem Description
Committed artifact trained with sklearn 1.9.1; installed environment uses sklearn 1.9.0.

### Resolution
Run `python -m src.train` to regenerate the artifact in the current environment.

---

## Summary Table

| Bug ID | File | Severity | Status |
|--------|------|----------|--------|
| BUG-001 | `src/classification.py` | Critical | Fixed |
| BUG-002 | `src/classification.py` | High | Fixed |
| BUG-003 | `src/train.py` | Medium | Fixed |
| BUG-004 | `src/advanced_features.py` | High | Fixed |
| BUG-005 | `src/advanced_features.py` | Medium | Fixed |
| BUG-006 | `src/evaluation.py` | Medium | Fixed |
| BUG-007 | `src/classification.py` | Medium | Fixed |
| BUG-008 | `src/train.py` | Low | Fixed |
| BUG-009 | `src/clustering.py` | Low | Fixed |
| BUG-010 | `requirements.txt` | Low | Fixed |
| BUG-011 | Committed artifact | Low | Run `python -m src.train` |
