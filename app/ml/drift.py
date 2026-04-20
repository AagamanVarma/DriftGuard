"""Drift baseline and detection utilities."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


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

    incoming_lengths = np.array([len(str(t)) for t in incoming_texts], dtype=float)
    incoming_mean_len = float(incoming_lengths.mean())
    length_shift = abs(incoming_mean_len - baseline_mean_len) / max(baseline_mean_len, 1.0)
    length_component = min(float(length_shift), 1.0)

    all_tokens: List[str] = []
    for text in incoming_texts:
        all_tokens.extend(_tokenize(text))

    if all_tokens and vocab:
        oov_tokens = sum(1 for tok in all_tokens if tok not in vocab)
        oov_rate = float(oov_tokens / len(all_tokens))
    elif all_tokens:
        oov_rate = 1.0
    else:
        oov_rate = 0.0

    oov_component = min(oov_rate * 2.0, 1.0)

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
