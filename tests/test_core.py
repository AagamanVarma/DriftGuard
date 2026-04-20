from pathlib import Path

import pytest

from app.ml.drift import detect_text_drift
from app.ml.metrics import weighted_score
from app.ml.train_utils import load_dataset


def test_detect_text_drift_empty_input_returns_no_drift():
    baseline = {
        "length_mean": 10.0,
        "length_std": 2.0,
        "length_samples": [8.0, 10.0, 12.0],
        "label_distribution": {"positive": 0.5, "negative": 0.5},
    }

    result = detect_text_drift(incoming_texts=[], baseline=baseline)

    assert result["is_drift"] is False
    assert result["drift_score"] == 0.0
    assert result["components"] == {
        "length_test": {"ks_statistic": 0.0, "p_value": 1.0},
        "label_test": None,
    }


def test_detect_text_drift_flags_length_distribution_shift():
    baseline = {
        "length_mean": 10.0,
        "length_std": 2.0,
        "length_samples": [8.0, 9.0, 10.0, 11.0, 12.0],
        "label_distribution": {"positive": 0.5, "negative": 0.5},
    }

    incoming_texts = ["x" * 80, "y" * 85, "z" * 90, "w" * 95]
    result = detect_text_drift(incoming_texts=incoming_texts, baseline=baseline, threshold=0.05)

    assert result["is_drift"] is True
    assert result["components"]["length_test"]["ks_statistic"] > 0.0
    assert result["components"]["length_test"]["p_value"] < 0.05


def test_detect_text_drift_flags_label_distribution_shift():
    baseline = {
        "length_mean": 10.0,
        "length_std": 2.0,
        "length_samples": [8.0, 9.0, 10.0, 11.0, 12.0],
        "label_distribution": {"positive": 0.8, "negative": 0.2},
    }

    incoming_texts = ["same size", "same size", "same size", "same size", "same size"]
    incoming_labels = ["negative", "negative", "negative", "negative", "positive"]

    result = detect_text_drift(
        incoming_texts=incoming_texts,
        baseline=baseline,
        incoming_labels=incoming_labels,
        threshold=0.05,
    )

    assert result["is_drift"] is True
    assert result["components"]["label_test"] is not None
    assert result["components"]["label_test"]["chi2_statistic"] > 0.0
    assert result["components"]["label_test"]["p_value"] < 0.05


def test_weighted_score_with_default_weights():
    metrics = {
        "accuracy": 0.9,
        "f1_score": 0.8,
        "precision": 0.7,
        "recall": 0.6,
    }

    score = weighted_score(metrics)

    expected = 0.9 * 0.4 + 0.8 * 0.4 + 0.7 * 0.1 + 0.6 * 0.1
    assert score == pytest.approx(expected)


def test_load_dataset_raises_for_missing_columns(tmp_path: Path):
    csv_path = tmp_path / "bad_dataset.csv"
    csv_path.write_text("id,text\n1,hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing columns"):
        load_dataset(csv_path)
