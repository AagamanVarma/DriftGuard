"""Training data prep, model training, and selection utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from app.ml.drift import build_drift_baseline
from app.ml.metrics import evaluate_predictions, measure_inference_latency, weighted_score
from app.ml.model_store import ModelStore


REQUIRED_COLUMNS = {"id", "text", "label"}

PARAM_CANDIDATES = {
    "LogisticRegression": [
        {"C": 1.0, "solver": "lbfgs"},
        {"C": 0.5, "solver": "liblinear"},
        {"C": 2.0, "solver": "lbfgs"},
    ],
    "RandomForest": [
        {"n_estimators": 120, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2},
        {"n_estimators": 180, "max_depth": 30, "min_samples_split": 4, "min_samples_leaf": 2},
        {"n_estimators": 100, "max_depth": 15, "min_samples_split": 6, "min_samples_leaf": 3},
    ],
    "LinearSVC": [
        {"C": 1.0, "loss": "squared_hinge"},
        {"C": 0.5, "loss": "squared_hinge"},
        {"C": 2.0, "loss": "squared_hinge"},
    ],
}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate dataset with required schema."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected {REQUIRED_COLUMNS}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    return df


def split_and_vectorize(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    max_features: int = 1000,
) -> Dict[str, Any]:
    """Split data and vectorize text with TF-IDF."""
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val["label"],
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        lowercase=True,
        min_df=2,
        max_df=0.95,
    )

    X_train = vectorizer.fit_transform(train["text"]).toarray()
    X_val = vectorizer.transform(val["text"]).toarray()
    X_test = vectorizer.transform(test["text"]).toarray()

    return {
        "vectorizer": vectorizer,
        "X_train": X_train,
        "y_train": train["label"].values,
        "X_val": X_val,
        "y_val": val["label"].values,
        "X_test": X_test,
        "y_test": test["label"].values,
        "class_distribution": df["label"].value_counts().to_dict(),
    }


def create_model(model_type: str, params: Dict[str, Any], random_state: int = 42) -> Any:
    """Factory for supported models."""
    if model_type == "LogisticRegression":
        return LogisticRegression(random_state=random_state, max_iter=1000, **params)
    if model_type == "RandomForest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)
    if model_type == "LinearSVC":
        return LinearSVC(random_state=random_state, dual=False, max_iter=2000, **params)
    raise ValueError(f"Unsupported model type: {model_type}")


def default_params(model_type: str) -> Dict[str, Any]:
    """Fallback params if optimization is off."""
    defaults = {
        "LogisticRegression": {"C": 1.0, "solver": "lbfgs"},
        "RandomForest": {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
        "LinearSVC": {"C": 1.0, "loss": "squared_hinge"},
    }
    return defaults[model_type]


def candidate_params(model_type: str, optimize: bool) -> List[Dict[str, Any]]:
    """Return readable candidate configs for each model."""
    if not optimize:
        return [default_params(model_type)]
    return PARAM_CANDIDATES[model_type]


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimize: bool = True,
    n_trials: int = 10,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Train all candidate models and return their validation metrics."""
    _ = n_trials
    results: Dict[str, Dict[str, Any]] = {}
    for model_type in ["LogisticRegression", "RandomForest", "LinearSVC"]:
        best_payload: Dict[str, Any] | None = None
        best_f1 = -1.0

        for params in candidate_params(model_type, optimize=optimize):
            model = create_model(model_type, params, random_state=random_state)
            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            latency = measure_inference_latency(model, X_val)
            metrics = evaluate_predictions(y_val, val_preds, latency)
            curr_f1 = float(metrics["f1_score"])

            if curr_f1 > best_f1:
                best_f1 = curr_f1
                best_payload = {
                    "model": model,
                    "metrics": metrics,
                    "config": {
                        "model_type": model_type,
                        "hyperparameters": params,
                        "feature_config": {"vectorizer": "TfidfVectorizer"},
                    },
                }

        if best_payload is None:
            raise RuntimeError(f"No model could be trained for {model_type}")

        results[model_type] = best_payload

    return results


def pick_best_model(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any], float]:
    """Pick best model by weighted score."""
    best_type, best_payload = max(
        results.items(),
        key=lambda item: weighted_score(item[1]["metrics"]),
    )
    score = weighted_score(best_payload["metrics"])
    return best_type, best_payload, score


def train_and_promote(
    dataset_path: Path,
    models_dir: Path,
    production_dir: Path,
    optimize: bool = True,
    n_trials: int = 10,
    random_state: int = 42,
    promote: bool = True,
) -> Dict[str, Any]:
    """End-to-end training pipeline that saves best model and optionally sets production."""
    df = load_dataset(dataset_path)
    packed = split_and_vectorize(df, test_size=0.15, val_size=0.15, random_state=random_state)

    trained_models = train_models(
        packed["X_train"],
        packed["y_train"],
        packed["X_val"],
        packed["y_val"],
        optimize=optimize,
        n_trials=n_trials,
        random_state=random_state,
    )

    test_results: Dict[str, Dict[str, Any]] = {}
    for model_type, payload in trained_models.items():
        model = payload["model"]
        y_pred = model.predict(packed["X_test"])
        test_metrics = evaluate_predictions(
            packed["y_test"],
            y_pred,
            latency_ms=measure_inference_latency(model, packed["X_test"]),
        )
        test_results[model_type] = {
            "model": model,
            "metrics": test_metrics,
            "config": payload["config"],
        }

    best_type, best_payload, best_score = pick_best_model(test_results)

    best_payload["config"]["drift_baseline"] = build_drift_baseline(
        texts=df["text"].astype(str).tolist(),
        labels=df["label"].astype(str).tolist(),
        vectorizer=packed["vectorizer"],
    )

    store = ModelStore(models_dir=models_dir, production_dir=production_dir)
    version = store.save(
        model=best_payload["model"],
        vectorizer=packed["vectorizer"],
        metrics=best_payload["metrics"],
        config=best_payload["config"],
    )
    if promote:
        store.set_production(version)

    return {
        "version": version,
        "promoted": bool(promote),
        "best_model_type": best_type,
        "best_score": float(best_score),
        "metrics": best_payload["metrics"],
        "config": best_payload["config"],
        "class_distribution": packed["class_distribution"],
    }
