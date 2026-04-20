"""Drift baseline and detection utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer


def build_drift_baseline(
    texts: List[str],
    labels: List[str],
    vectorizer: TfidfVectorizer,
) -> Dict[str, Any]:
    """Create baseline statistics from training data for future drift checks."""
    if not texts:
        raise ValueError("Cannot build drift baseline from empty text list")

    lengths = np.array([len(str(t)) for t in texts], dtype=float)

    label_series = pd.Series(labels, dtype="string")
    label_dist = (label_series.value_counts(normalize=True)).to_dict()
    label_dist = {str(k): float(v) for k, v in label_dist.items()}

    return {
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std(ddof=0)),
        "length_samples": [float(length) for length in lengths.tolist()],
        "label_distribution": label_dist,
        "created_at": datetime.utcnow().isoformat(),
    }


def detect_text_drift(
    incoming_texts: List[str],
    baseline: Dict[str, Any],
    incoming_labels: List[str] | None = None,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """Test drift from incoming data against saved baseline statistics.

    Uses a two-sample Kolmogorov-Smirnov test for text lengths and a
    chi-square goodness-of-fit test for label distribution shift when labels
    are available. The threshold is interpreted as the significance level
    (alpha) for both tests.
    """
    if not incoming_texts:
        return {
            "is_drift": False,
            "drift_score": 0.0,
            "threshold": float(threshold),
            "components": {
                "length_test": {"ks_statistic": 0.0, "p_value": 1.0},
                "label_test": None,
            },
        }

    baseline_lengths = np.array(baseline.get("length_samples", []), dtype=float)
    incoming_lengths = np.array([len(str(t)) for t in incoming_texts], dtype=float)

    if baseline_lengths.size == 0:
        ks_statistic = 0.0
        p_value = 1.0
    else:
        ks_result = ks_2samp(baseline_lengths, incoming_lengths, alternative="two-sided", mode="auto")
        ks_statistic = float(ks_result.statistic)
        p_value = float(ks_result.pvalue)

    label_test = None
    label_p_value = 1.0
    baseline_label_dist = baseline.get("label_distribution", {})
    if incoming_labels and baseline_label_dist:
        incoming_label_counts = pd.Series(incoming_labels, dtype="string").value_counts().to_dict()
        label_keys = sorted(set(incoming_label_counts) | set(baseline_label_dist))
        observed = np.array([float(incoming_label_counts.get(label, 0)) for label in label_keys], dtype=float)
        expected_probs = np.array([float(baseline_label_dist.get(label, 0.0)) for label in label_keys], dtype=float)

        if observed.sum() > 0 and expected_probs.sum() > 0:
            expected = expected_probs / expected_probs.sum() * observed.sum()
            epsilon = 1e-12
            expected = np.where(expected <= 0.0, epsilon, expected)
            expected *= observed.sum() / expected.sum()
            chi2_result = chisquare(f_obs=observed, f_exp=expected)
            label_p_value = float(chi2_result.pvalue)
            label_test = {
                "chi2_statistic": float(chi2_result.statistic),
                "p_value": label_p_value,
            }

    is_length_drift = p_value < threshold
    is_label_drift = label_test is not None and label_p_value < threshold

    return {
        "is_drift": bool(is_length_drift or is_label_drift),
        "drift_score": float(max(1.0 - p_value, 1.0 - label_p_value)),
        "threshold": float(threshold),
        "components": {
            "length_test": {"ks_statistic": ks_statistic, "p_value": p_value},
            "label_test": label_test,
        },
        "incoming_summary": {
            "num_samples": int(len(incoming_texts)),
            "mean_text_length": float(incoming_lengths.mean()),
            "label_count": int(len(incoming_labels)) if incoming_labels else 0,
        },
    }
