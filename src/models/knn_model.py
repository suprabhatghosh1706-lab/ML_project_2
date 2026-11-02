"""K-Nearest Neighbors classifier pipeline for fraud detection."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_knn_model(
    n_neighbors: int = 5,
    weights: str = "distance",
    imputer_strategy: str = "median",
) -> Pipeline:
    """Create and return a K-Nearest Neighbors pipeline.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for kneighbors queries.
    weights : str, default="distance"
        Weight function used in prediction ("uniform", "distance").
    imputer_strategy : str, default="median"
        Strategy for SimpleImputer ("mean", "median", etc.).

    Returns
    -------
    Pipeline
        Configured Pipeline containing imputer, scaler, and KNeighborsClassifier.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
    ])
