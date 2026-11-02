"""Single Layer Perceptron (SLP) pipeline for fraud detection."""
from __future__ import annotations

from typing import Any
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_perceptron_model(
    class_weight: str | dict[Any, Any] | None = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
    imputer_strategy: str = "median",
) -> Pipeline:
    """Create and return a Single Layer Perceptron pipeline.

    Parameters
    ----------
    class_weight : str, dict, or None, default="balanced"
        Preset for the class_weight fit parameter.
    max_iter : int, default=1000
        The maximum number of passes over the training data.
    random_state : int, default=42
        Seed used to shuffle the data.
    imputer_strategy : str, default="median"
        Strategy for SimpleImputer ("mean", "median", etc.).

    Returns
    -------
    Pipeline
        Configured Pipeline containing imputer, scaler, and Perceptron.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", StandardScaler()),
        ("model", Perceptron(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )),
    ])
