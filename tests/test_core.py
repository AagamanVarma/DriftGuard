from pathlib import Path

import pytest

from app.core import detect_text_drift, load_dataset, weighted_score


def test_detect_text_drift_empty_input_returns_no_drift():
    baseline = {
        "length_mean": 10.0,
        "label_distribution": {"positive": 0.5, "negative": 0.5},
        "vocabulary": ["good", "bad"],
    }

    result = detect_text_drift(incoming_texts=[], baseline=baseline)

    assert result["is_drift"] is False
    assert result["drift_score"] == 0.0
    assert result["components"] == {
        "length_shift": 0.0,
        "oov_rate": 0.0,
        "label_shift": 0.0,
    }


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
