"""Dummy baseline classifier for fraud detection."""
from __future__ import annotations

from typing import Any
from sklearn.dummy import DummyClassifier


def create_dummy_model(
    strategy: str = "most_frequent",
    random_state: int | None = None,
    constant: Any = None,
) -> DummyClassifier:
    """Create and return a DummyClassifier baseline.

    Parameters
    ----------
    strategy : str, default="most_frequent"
        Strategy to use for generating predictions.
        Options: "most_frequent", "prior", "stratified", "uniform", "constant".
    random_state : int or None, default=None
        Controls randomness if applicable.
    constant : Any, default=None
        The explicit constant prediction if strategy="constant".

    Returns
    -------
    DummyClassifier
        Configured scikit-learn DummyClassifier estimator.
    """
    kwargs: dict[str, Any] = {"strategy": strategy}
    if random_state is not None:
        kwargs["random_state"] = random_state
    if constant is not None:
        kwargs["constant"] = constant
    return DummyClassifier(**kwargs)
