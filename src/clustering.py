"""Standalone clustering exploratory script.

Runs K-Means (k=2) and DBSCAN on a sample of the dataset and saves a
side-by-side PCA scatter plot.  This is exploratory analysis only;
cluster assignments do NOT prove fraud.

Usage
-----
    python -m src.clustering
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
REPORTS = ROOT / "reports"

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Place creditcard.csv in {DATA_PATH.parent}\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )

    REPORTS.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

    # Select numeric features only; exclude the target.
    feature_cols = [
        col
        for col in sample.columns
        if col != "Class" and pd.api.types.is_numeric_dtype(sample[col])
    ]

    if len(feature_cols) < 2:
        raise ValueError(
            f"At least two numeric feature columns are required. "
            f"Found: {feature_cols}"
        )

    X = sample[feature_cols].copy().replace(
        [float("inf"), float("-inf")], float("nan")
    )

    # Drop rows with any missing values after replacement.
    valid_mask = X.notna().all(axis=1)
    X = X.loc[valid_mask]
    y = sample.loc[valid_mask, "Class"]

    if len(X) < 10:
        raise ValueError(
            "Not enough complete numeric rows for clustering "
            f"(got {len(X)}, need at least 10)."
        )

    X_scaled = StandardScaler().fit_transform(X)

    n_components = min(2, X_scaled.shape[1])
    coords = PCA(
        n_components=n_components, random_state=RANDOM_STATE
    ).fit_transform(X_scaled)

    if n_components == 1:
        coords = np.column_stack([coords, np.zeros(len(coords))])

    km_labels = KMeans(
        n_clusters=2, n_init=10, random_state=RANDOM_STATE
    ).fit_predict(X_scaled)

    db_labels = DBSCAN(eps=3.0, min_samples=10, n_jobs=-1).fit_predict(X_scaled)

    print("K-Means cluster sizes:")
    print(pd.Series(km_labels).value_counts().sort_index())
    print("\nDBSCAN labels (-1 means noise):")
    print(pd.Series(db_labels).value_counts().sort_index())
    print(
        "\nExploratory agreement with known Class labels "
        "(not a fraud accuracy score):"
    )
    print(
        "K-Means adjusted rand index:",
        round(adjusted_rand_score(y, km_labels), 4),
    )
    print(
        "DBSCAN adjusted rand index:",
        round(adjusted_rand_score(y, db_labels), 4),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(coords[:, 0], coords[:, 1], c=km_labels, s=5, cmap="tab10")
    axes[0].set_title("K-Means clusters (PCA view)")
    axes[1].scatter(coords[:, 0], coords[:, 1], c=db_labels, s=5, cmap="tab10")
    axes[1].set_title("DBSCAN clusters / noise (PCA view)")
    for ax in axes:
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
    fig.suptitle(
        "Exploratory clustering — clusters do not represent confirmed fraud",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = REPORTS / "clustering_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nGraph saved to {out_path}")


if __name__ == "__main__":
    main()
