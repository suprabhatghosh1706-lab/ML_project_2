"""Random Forest classifier pipeline for fraud detection."""
from __future__ import annotations

from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def create_random_forest_model(
    n_estimators: int = 100,
    class_weight: str | dict[Any, Any] | None = "balanced_subsample",
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    random_state: int = 42,
    imputer_strategy: str = "median",
) -> Pipeline:
    """Create and return a Random Forest pipeline with median imputation.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    class_weight : str, dict, or None, default="balanced_subsample"
        Weights associated with classes.
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples_leaf : int, default=2
        The minimum number of samples required to be at a leaf node.
    n_jobs : int, default=-1
        The number of jobs to run in parallel (-1 uses all available cores).
    random_state : int, default=42
        Controls the randomness of bootstrapping and feature selection.
    imputer_strategy : str, default="median"
        Strategy for SimpleImputer ("mean", "median", etc.).

    Returns
    -------
    Pipeline
        Configured Pipeline containing imputer and RandomForestClassifier.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )),
    ])
