"""Model evaluation and ranking metrics."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_WEIGHTS = {
    "accuracy": 0.4,
    "f1_score": 0.4,
    "precision": 0.1,
    "recall": 0.1,
}


def measure_inference_latency(model: Any, X: np.ndarray, num_runs: int = 100) -> float:
    """Average latency in milliseconds.

    Supports dense numpy arrays and scipy sparse matrices.
    """
    if hasattr(X, "shape") and X.shape is not None:
        rows = int(X.shape[0])
    else:
        rows = int(len(X))
    X_sample = X[:num_runs] if rows > num_runs else X
    start = time.perf_counter()
    for _ in range(num_runs):
        model.predict(X_sample)
    end = time.perf_counter()
    return float((end - start) / num_runs * 1000)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, latency_ms: float) -> Dict[str, Any]:
    """Compute standard metrics and return as dict."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "latency_ms": float(latency_ms),
        "timestamp": datetime.utcnow().isoformat(),
    }


def weighted_score(metrics: Dict[str, Any], weights: Dict[str, float] | None = None) -> float:
    """Simple weighted ranking score."""
    weights = weights or DEFAULT_WEIGHTS
    total = sum(weights.values())
    normalized = {k: v / total for k, v in weights.items()}

    return float(
        metrics["accuracy"] * normalized["accuracy"]
        + metrics["f1_score"] * normalized["f1_score"]
        + metrics["precision"] * normalized["precision"]
        + metrics["recall"] * normalized["recall"]
    )
