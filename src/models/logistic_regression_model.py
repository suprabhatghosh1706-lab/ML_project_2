"""Logistic Regression model pipeline for fraud detection."""
from __future__ import annotations

from typing import Any
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_logistic_regression_model(
    class_weight: str | dict[Any, Any] | None = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
    solver: str = "lbfgs",
    imputer_strategy: str = "median",
    C: float = 1.0,
) -> Pipeline:
    """Create and return a Logistic Regression pipeline with imputation and scaling.

    Parameters
    ----------
    class_weight : str, dict, or None, default="balanced"
        Weights associated with classes in the form {class_label: weight}.
    max_iter : int, default=1000
        Maximum number of iterations taken for the solvers to converge.
    random_state : int, default=42
        Seed used by the random number generator.
    solver : str, default="lbfgs"
        Algorithm to use in the optimization problem.
    imputer_strategy : str, default="median"
        Strategy for SimpleImputer ("mean", "median", etc.).
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.

    Returns
    -------
    Pipeline
        Configured Pipeline containing imputer, scaler, and LogisticRegression model.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            solver=solver,
            C=C,
        )),
    ])
