# Corrections made

1. The project had two training scripts writing to the same `artifacts/fraud_model.joblib`. Running the course-algorithm comparison could replace the dashboard model with a different model that had no threshold metadata. The comparison script now saves `artifacts/comparison_model.joblib`; only `src.train` creates the dashboard model.
2. The dashboard now requires a saved threshold and warns when the model name or threshold does not match the metrics report. This prevents old model results from being shown as if they belong to the current model.
3. Every upload now gets positional `source_row_number` values starting at 1 and an upload-run ID based on file contents, model, and threshold. The individual risk explorer and downloaded scored CSV use the same scored dataframe.
4. The alert history now records source file, source row, threshold, and upload-run ID. Saving the same upload at the same threshold does not append duplicate alerts. Old history files without a run ID are moved aside as `fraud_alert_history_legacy_backup.csv` rather than mixed with new records.
5. Historical alerts are explicitly shown as history; they are not treated as the current upload's predictions. This addresses confusing serial numbers and amounts from older uploads.
6. The model-comparison chart recognises the validation metric column names produced by the canonical training script.

## Required first run after replacing the project

Open the terminal in the project root and run:

```powershell
python -m pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

The tree models use 100 estimators to reduce training time. The provided old model and generated reports were intentionally not carried into this corrected package because they came from different training scripts and were not a consistent set. Training recreates the model and reports together. The existing legacy alert history is not included in this clean package; if using the corrected code in the old project folder, the dashboard automatically moves a legacy history file aside on first use.
