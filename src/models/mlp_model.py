"""Multi-Layer Perceptron (MLP) neural network pipeline for fraud detection."""
from __future__ import annotations

from typing import Sequence
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_mlp_model(
    hidden_layer_sizes: Sequence[int] = (32, 16),
    max_iter: int = 30,
    early_stopping: bool = True,
    random_state: int = 42,
    imputer_strategy: str = "median",
) -> Pipeline:
    """Create and return a Multi-Layer Perceptron pipeline.

    Parameters
    ----------
    hidden_layer_sizes : Sequence[int], default=(32, 16)
        The ith element represents the number of neurons in the ith hidden layer.
    max_iter : int, default=30
        Maximum number of iterations.
    early_stopping : bool, default=True
        Whether to use early stopping when validation score is not improving.
    random_state : int, default=42
        Determines random number generation for weights and bias initialization.
    imputer_strategy : str, default="median"
        Strategy for SimpleImputer ("mean", "median", etc.).

    Returns
    -------
    Pipeline
        Configured Pipeline containing imputer, scaler, and MLPClassifier.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )),
    ])
