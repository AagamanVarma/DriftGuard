"""Simple core utilities for training, evaluation, and model storage."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


REQUIRED_COLUMNS = {"id", "text", "label"}
DEFAULT_WEIGHTS = {
    "accuracy": 0.4,
    "f1_score": 0.4,
    "precision": 0.1,
    "recall": 0.1,
}

# We keep a small, readable set of parameter candidates per model.
# This gives better results than one fixed config, but is still easy to explain.
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


def setup_logger(name: str) -> logging.Logger:
    """Create a basic console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


class ModelStore:
    """Very simple file-based model store with production pointer."""

    def __init__(self, models_dir: Path, production_dir: Path):
        self.models_dir = Path(models_dir)
        self.production_dir = Path(production_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.production_dir.mkdir(parents=True, exist_ok=True)
        self.current_model_file = self.production_dir / "current_model.txt"

    def get_next_version(self) -> str:
        versions: List[int] = []
        for d in self.models_dir.iterdir():
            if d.is_dir() and d.name.startswith("model_v"):
                try:
                    versions.append(int(d.name.replace("model_v", "")))
                except ValueError:
                    continue
        return f"model_v{(max(versions) + 1) if versions else 1}"

    def save(
        self,
        model: Any,
        vectorizer: TfidfVectorizer,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        version: str | None = None,
    ) -> str:
        version = version or self.get_next_version()
        model_dir = self.models_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_dir / "model.pkl")
        joblib.dump(vectorizer, model_dir / "vectorizer.pkl")

        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return version

    def exists(self, version: str) -> bool:
        return (self.models_dir / version / "model.pkl").exists()

    def load_model(self, version: str) -> Any:
        return joblib.load(self.models_dir / version / "model.pkl")

    def load_vectorizer(self, version: str) -> TfidfVectorizer:
        return joblib.load(self.models_dir / version / "vectorizer.pkl")

    def load_metrics(self, version: str) -> Dict[str, Any]:
        with open(self.models_dir / version / "metrics.json", "r") as f:
            return json.load(f)

    def load_config(self, version: str) -> Dict[str, Any]:
        with open(self.models_dir / version / "config.json", "r") as f:
            return json.load(f)

    def set_production(self, version: str) -> None:
        if not self.exists(version):
            raise FileNotFoundError(f"Model {version} not found")
        with open(self.current_model_file, "w") as f:
            f.write(version)

    def get_production(self) -> str:
        if not self.current_model_file.exists():
            raise FileNotFoundError("No production model set")
        with open(self.current_model_file, "r") as f:
            return f.read().strip()

    def list_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        try:
            production_version = self.get_production()
        except FileNotFoundError:
            production_version = None

        for d in sorted(self.models_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("model_v"):
                continue
            try:
                models.append(
                    {
                        "version": d.name,
                        "is_production": d.name == production_version,
                        "metrics": self.load_metrics(d.name),
                        "config": self.load_config(d.name),
                    }
                )
            except Exception:
                continue
        return models


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
    # 1) Keep a final test split untouched for fair model comparison.
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    # 2) Split remaining data into train and validation.
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


def measure_inference_latency(model: Any, X: np.ndarray, num_runs: int = 100) -> float:
    """Average latency in milliseconds."""
    X_sample = X[:num_runs] if len(X) > num_runs else X
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


def _tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenizer."""
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def build_drift_baseline(
    texts: List[str],
    labels: List[str],
    vectorizer: TfidfVectorizer,
) -> Dict[str, Any]:
    """Create baseline statistics from training data for future drift checks."""
    if not texts:
        raise ValueError("Cannot build drift baseline from empty text list")

    # These simple summary stats are cheap and easy to explain in viva/interviews.
    lengths = np.array([len(str(t)) for t in texts], dtype=float)
    token_counts = np.array([len(_tokenize(t)) for t in texts], dtype=float)
    vocab = [str(v) for v in vectorizer.get_feature_names_out()]

    label_series = pd.Series(labels, dtype="string")
    label_dist = (label_series.value_counts(normalize=True)).to_dict()
    label_dist = {str(k): float(v) for k, v in label_dist.items()}

    return {
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std(ddof=0)),
        "avg_tokens": float(token_counts.mean()),
        "label_distribution": label_dist,
        "vocabulary": vocab,
        "created_at": datetime.utcnow().isoformat(),
    }


def detect_text_drift(
    incoming_texts: List[str],
    baseline: Dict[str, Any],
    incoming_labels: List[str] | None = None,
    threshold: float = 0.35,
) -> Dict[str, Any]:
    """Score drift from incoming texts against saved baseline statistics."""
    if not incoming_texts:
        return {
            "is_drift": False,
            "drift_score": 0.0,
            "threshold": float(threshold),
            "components": {"length_shift": 0.0, "oov_rate": 0.0, "label_shift": 0.0},
        }

    baseline_mean_len = float(baseline.get("length_mean", 1.0))
    baseline_labels = baseline.get("label_distribution", {})
    vocab = set(str(v) for v in baseline.get("vocabulary", []))

    # Component 1: text length shift (are texts becoming much longer/shorter?)
    incoming_lengths = np.array([len(str(t)) for t in incoming_texts], dtype=float)
    incoming_mean_len = float(incoming_lengths.mean())
    length_shift = abs(incoming_mean_len - baseline_mean_len) / max(baseline_mean_len, 1.0)
    length_component = min(float(length_shift), 1.0)

    all_tokens: List[str] = []
    for text in incoming_texts:
        all_tokens.extend(_tokenize(text))

    # Component 2: out-of-vocabulary rate (how many new words are unseen?)
    if all_tokens and vocab:
        oov_tokens = sum(1 for tok in all_tokens if tok not in vocab)
        oov_rate = float(oov_tokens / len(all_tokens))
    elif all_tokens:
        oov_rate = 1.0
    else:
        oov_rate = 0.0

    oov_component = min(oov_rate * 2.0, 1.0)

    # Component 3: label distribution shift (only when labels are provided)
    label_component = 0.0
    if incoming_labels:
        inc_dist_raw = pd.Series(incoming_labels, dtype="string").value_counts(normalize=True).to_dict()
        inc_dist = {str(k): float(v) for k, v in inc_dist_raw.items()}
        keys = set(inc_dist) | set(baseline_labels)
        label_component = 0.5 * sum(abs(inc_dist.get(k, 0.0) - float(baseline_labels.get(k, 0.0))) for k in keys)

    components = {
        "length_shift": float(length_component),
        "oov_rate": float(oov_component),
        "label_shift": float(min(label_component, 1.0)),
    }

    weights = {"length_shift": 0.5, "oov_rate": 0.3, "label_shift": 0.2}
    if incoming_labels is None:
        weights = {"length_shift": 0.625, "oov_rate": 0.375, "label_shift": 0.0}

    drift_score = (
        components["length_shift"] * weights["length_shift"]
        + components["oov_rate"] * weights["oov_rate"]
        + components["label_shift"] * weights["label_shift"]
    )

    return {
        "is_drift": bool(drift_score >= threshold),
        "drift_score": float(drift_score),
        "threshold": float(threshold),
        "components": components,
        "incoming_summary": {
            "num_samples": int(len(incoming_texts)),
            "mean_text_length": float(incoming_mean_len),
            "token_count": int(len(all_tokens)),
        },
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
    results: Dict[str, Dict[str, Any]] = {}
    for model_type in ["LogisticRegression", "RandomForest", "LinearSVC"]:
        # For each algorithm, pick the best config based on validation F1.
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
    # Step A: prepare data
    df = load_dataset(dataset_path)
    packed = split_and_vectorize(df, test_size=0.15, val_size=0.15, random_state=random_state)

    # Step B: train candidate algorithms
    trained_models = train_models(
        packed["X_train"],
        packed["y_train"],
        packed["X_val"],
        packed["y_val"],
        optimize=optimize,
        n_trials=n_trials,
        random_state=random_state,
    )

    # Step C: evaluate each trained algorithm on the held-out test split
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

    # Step D: select winner and store baseline stats for drift checks
    best_type, best_payload, best_score = pick_best_model(test_results)

    best_payload["config"]["drift_baseline"] = build_drift_baseline(
        texts=df["text"].astype(str).tolist(),
        labels=df["label"].astype(str).tolist(),
        vectorizer=packed["vectorizer"],
    )

    # Step E: save selected model and optionally promote as production
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
