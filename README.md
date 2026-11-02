# Credit Card Fraud Detection Using Machine Learning

A student-project prototype based on the ULB-style anonymised credit-card dataset. It inspects data, trains and compares four classifiers, evaluates the selected model, saves the fitted pipeline and provides a Streamlit dashboard for batch CSV scoring.

## Features

- Dataset validation and exploratory data analysis (EDA)
- Stratified train / validation / test split (70 / 15 / 15)
- Dummy baseline, Logistic Regression, Random Forest and Extra Trees for the dashboard model
- Separate optional comparison of Logistic Regression, KNN, Naive Bayes, Perceptron, MLP and Random Forest
- Class weighting for imbalance; no resampling applied to validation or test data
- Validation PR-AUC comparison and validation-based threshold selection (F1-maximising)
- Fraud class index derived from `model.classes_` — no assumption that column 1 is always fraud
- Precision, recall, F1, PR-AUC, ROC-AUC, and confusion-matrix counts
- Saved model with `joblib` (includes threshold, validation metrics, fraud class label, model version)
- Streamlit CSV upload, predictions, adjustable threshold, risk levels, and CSV download
- Fraud alert history with duplicate prevention and legacy file separation
- K-Means and DBSCAN clustering explorer (exploratory only)
- Automated test suite using `pytest`

## 1. Requirements

- Python 3.10 or newer recommended
- VS Code or another Python editor
- The public dataset CSV

## 2. Put the dataset in the project

Download the public Credit Card Fraud Detection dataset from Kaggle. Put the CSV here:

`data/creditcard.csv`

The expected columns are `Time`, `V1` through `V28`, `Amount`, and `Class`. The target should be `Class = 1` for fraud and `Class = 0` for legitimate transactions. Check the downloaded dataset's current terms / licence. Do not use real cardholder details.

## 3. Create a virtual environment (recommended)

Open a terminal in this project folder.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and `.venv\Scripts\activate.bat`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run EDA (optional)

```bash
python -m src.eda
```

Prints dataset shape, column types, missing values, duplicate count and class balance. Saves plots in `reports/`.

## 5. Train and evaluate the models

```bash
python -m src.train
```

Creates the canonical dashboard model and matching reports:
- `artifacts/fraud_model.joblib`
- `reports/model_metrics.json`
- `reports/model_comparison.csv`

Training can take several minutes depending on the computer. Tree models use 100 estimators.

> **Do not report performance values before running this script and recording your actual results.**

### Optional course-algorithm comparison

Run `python -m src.classification` to compare the additional algorithms (Logistic Regression, KNN, Naive Bayes, Perceptron, MLP, Random Forest). This saves its own artifact as `artifacts/comparison_model.joblib` and **does not replace the dashboard model**. Metrics are computed on the validation set; the test set is not touched.

### Optional clustering

```bash
python -m src.clustering
```

Runs K-Means and DBSCAN on a sample and saves a side-by-side PCA scatter plot. Cluster assignments are exploratory and do not prove fraud.

## 6. Run evaluation summary

```bash
python -m src.evaluation
```

Reads `reports/model_comparison.csv` (or `reports/classification_comparison.csv` as a fallback) and `reports/model_metrics.json`, prints a formatted summary, and saves a bar chart.

## 7. Run automated tests

```bash
python -m pytest tests/test_project.py -v
```

Uses small synthetic datasets — no download required for unit tests.

## 8. Launch the dashboard

After training:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. Upload a CSV containing the same feature columns as the training dataset. The `Class` target is optional for scoring and is never passed as a model input. If you upload a file with trusted `Class` labels, the dashboard can also calculate evaluation counts and metrics.

## Project structure

```text
credit_card_fraud_detection/
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── README.md
├── BUG_REPORT.md             # Documented bugs and fixes
├── CORRECTIONS.md            # Historical corrections
├── .gitignore
├── data/
│   └── creditcard.csv        # Place dataset here
├── src/
│   ├── __init__.py
│   ├── common.py             # Shared paths and validation helpers
│   ├── eda.py                # Exploratory data analysis
│   ├── train.py              # Main training pipeline (dashboard model)
│   ├── classification.py     # Optional multi-algorithm comparison
│   ├── evaluation.py         # Evaluation summary and charts
│   ├── clustering.py         # Standalone K-Means / DBSCAN script
│   ├── dashboard_extras.py   # Analytics, filters, confusion matrix
│   └── advanced_features.py  # Visualisations, clustering UI, alert history
├── artifacts/
│   └── fraud_model.joblib    # Dashboard model (generated by src.train)
├── reports/
│   ├── model_metrics.json    # Final test metrics (generated by src.train)
│   ├── model_comparison.csv  # Validation comparison (generated by src.train)
│   └── fraud_alert_history.csv
└── tests/
    └── test_project.py       # Automated test suite
```

## Dataset requirements

| Column | Type | Description |
|---|---|---|
| `Time` | float | Seconds elapsed since the first transaction |
| `V1`–`V28` | float | PCA-anonymised features |
| `Amount` | float | Transaction amount |
| `Class` | int | 0 = legitimate, 1 = fraud (required for training; optional for scoring) |

## ML pipeline overview

```
Dataset
  └─ Validation (common.py)
       └─ Stratified 70/15/15 split
            ├─ Training set (70%)  → fit imputer, scaler, classifier
            ├─ Validation set (15%) → compare models, choose threshold
            └─ Test set (15%)      → final evaluation (once, after selection)
                   └─ Save artifact + reports
                          └─ Streamlit dashboard
                               └─ Upload CSV → score → alert history
```

## Important methodology notes

- The split is performed before model fitting.
- Imputation and scaling are fitted within each model pipeline on training data only.
- The final test split is not used to select the model or tune the threshold.
- Models are compared using validation PR-AUC; the selected model's threshold is chosen on validation data by maximising F1-score.
- The fraud class index is derived from `model.classes_`, not assumed to be column 1.
- Class weighting is the first imbalance-handling method; SMOTE is not enabled.
- The `Class` column is a label, not a feature — target leakage is explicitly prevented.
- Accuracy is shown only as a supporting metric because class imbalance makes it misleading.
- An output score is not necessarily a calibrated real-world fraud probability.
- This is an educational prototype, not a production banking system. Predictions should support human review, not automatically accuse a cardholder.

## Troubleshooting

| Symptom | Solution |
|---|---|
| `Dataset not found` | Check path `data/creditcard.csv` |
| `Missing 'Class'` | Ensure you downloaded the correct dataset |
| `Missing feature columns in dashboard` | Upload data with the same feature schema as training |
| `Threshold not in bundle` | Run `python -m src.train` to rebuild the dashboard model |
| `InconsistentVersionWarning` on model load | Run `python -m src.train` to regenerate with the current sklearn |
| `Old alert history` | Dashboard separates legacy history automatically |
| `Module import error` | Run commands from the project root (where `README.md` and `app.py` are) |
| `Training memory / time issues` | Reduce tree count in `src/train.py` or run in Google Colab |

## Outputs for the project report

Use your actual generated comparison table, metrics JSON, confusion-matrix counts, and EDA plots. Discuss:
- False positives vs false negatives and their business costs
- Class imbalance and how it affects accuracy-based metrics
- Dataset age, anonymisation, and limitations on real-world applicability
- Benchmark results do not guarantee real-world banking performance
