"""Backward-compatible facade for core ML utilities.

This module re-exports functions/classes from the newer split module layout
so older imports (e.g. `from app.core import ...`) keep working.
"""

from app.ml.drift import build_drift_baseline, detect_text_drift
from app.ml.metrics import DEFAULT_WEIGHTS, evaluate_predictions, measure_inference_latency, weighted_score
from app.ml.model_store import ModelStore
from app.ml.train_utils import (
    PARAM_CANDIDATES,
    REQUIRED_COLUMNS,
    candidate_params,
    create_model,
    default_params,
    load_dataset,
    pick_best_model,
    split_and_vectorize,
    train_and_promote,
    train_models,
)
from app.utils.logging import setup_logger

__all__ = [
    "ModelStore",
    "REQUIRED_COLUMNS",
    "DEFAULT_WEIGHTS",
    "PARAM_CANDIDATES",
    "setup_logger",
    "load_dataset",
    "split_and_vectorize",
    "measure_inference_latency",
    "evaluate_predictions",
    "build_drift_baseline",
    "detect_text_drift",
    "weighted_score",
    "create_model",
    "default_params",
    "candidate_params",
    "train_models",
    "pick_best_model",
    "train_and_promote",
]
