"""Gaussian Naive Bayes classifier pipeline for fraud detection."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


def create_naive_bayes_model(
    var_smoothing: float = 1e-9,
    imputer_strategy: str = "median",
) -> Pipeline:
    """Create and return a Gaussian Naive Bayes pipeline.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to variances for calculation stability.
    imputer_strategy : str, default="median"
        Strategy for SimpleImputer ("mean", "median", etc.).

    Returns
    -------
    Pipeline
        Configured Pipeline containing imputer and GaussianNB.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("model", GaussianNB(var_smoothing=var_smoothing)),
    ])
