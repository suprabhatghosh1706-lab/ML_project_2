"""Shared paths and dataset validation helpers."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = ARTIFACT_DIR / "fraud_model.joblib"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"

TARGET = "Class"

def load_dataset(path=DATA_PATH):
    """Load and validate the expected ULB-style dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download creditcard.csv and place it inside the project's data folder."
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError("The CSV must contain a 'Class' target column.")
    if df.empty:
        raise ValueError("The dataset is empty.")
    if df[TARGET].isna().any():
        raise ValueError("The target column 'Class' contains missing values.")
    labels = set(df[TARGET].unique())
    if not labels.issubset({0, 1, 0.0, 1.0}) or len(labels) < 2:
        raise ValueError("Expected Class labels 0 (legitimate) and 1 (fraud), both present.")
    return df

def feature_columns(df):
    """Return model inputs in their original CSV order, excluding the target."""
    return [column for column in df.columns if column != TARGET]


def validate_uploaded_dataframe(
    df: pd.DataFrame,
    required_features: list[str],
) -> tuple[bool, str, pd.DataFrame | None]:
    """Validate uploaded transaction dataframe against expected schema and constraints.

    Returns:
        tuple (is_valid: bool, error_message: str, cleaned_features_df: pd.DataFrame | None)
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, "No data provided or invalid file format.", None

    if df.empty:
        return False, "The uploaded CSV file is empty.", None

    # Check for duplicate column names in the CSV
    if len(df.columns) != len(set(df.columns)):
        dupes = [col for col in df.columns if list(df.columns).count(col) > 1]
        return False, f"Invalid CSV. Duplicate columns detected: {', '.join(sorted(set(dupes)))}", None

    # Required features must all be present (excluding target Class)
    expected = [f for f in required_features if f != TARGET]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        formatted_missing = "\n".join(f"- {col}" for col in missing)
        return False, (
            f"Invalid CSV.\nMissing required features:\n{formatted_missing}\n\n"
            "Please upload a CSV containing all required model features (Time, V1-V28, Amount)."
        ), None

    # Extract required features in strict canonical training order
    X = df[expected].copy()

    # Convert to numeric and identify non-numeric columns
    non_numeric_cols = []
    for col in X.columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if not pd.api.types.is_numeric_dtype(X[col]) and converted.isna().sum() > X[col].isna().sum():
            non_numeric_cols.append(col)
        X[col] = converted

    if non_numeric_cols:
        return False, (
            f"Invalid CSV data types.\nThe following feature columns contain non-numeric text values:\n"
            + ", ".join(non_numeric_cols)
        ), None

    # Replace inf with nan
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    # Missing values check
    missing_counts = X.isna().sum()
    cols_with_na = missing_counts[missing_counts > 0]
    if not cols_with_na.empty:
        details = ", ".join(f"{col} ({count} rows)" for col, count in cols_with_na.items())
        return False, (
            f"Invalid CSV data values.\nMissing, infinite, or corrupted values found in: {details}.\n"
            "Please provide a clean dataset with complete numeric entries."
        ), None

    return True, "", X
