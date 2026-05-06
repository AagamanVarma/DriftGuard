from pathlib import Path

from app.db.sqlite_store import SQLiteStore


def test_sqlite_store_stats_counts_rows(tmp_path: Path):
    db_path = tmp_path / "driftguard.db"
    store = SQLiteStore(db_path)
    store.init_db()

    # Add one row to each tracked table.
    store.log_prediction(
        request_text="great product",
        prediction="positive",
        probability=0.91,
        model_version="model_v1",
        probabilities={"positive": 0.91, "negative": 0.09},
    )

    batch_id = store.log_ingest_batch(
        num_records=1,
        min_window_size=1,
        drift_threshold=0.05,
        auto_retrain=False,
        drift_score=0.0,
        is_drift=False,
        retrained=False,
        promoted=None,
        champion_version="model_v1",
        challenger_version=None,
        new_version=None,
        message="No drift",
        drift_details={"is_drift": False, "drift_score": 0.0},
    )
    store.log_ingest_records(batch_id, [{"id": 1, "text": "sample", "label": "positive"}])

    store.upsert_model(
        version="model_v1",
        metrics={"accuracy": 0.9},
        config={"name": "baseline"},
        promoted=True,
    )

    stats = store.get_stats()

    assert stats["db_exists"] is True
    assert stats["db_size_bytes"] > 0
    assert stats["tables"]["predictions"] == 1
    assert stats["tables"]["ingest_batches"] == 1
    assert stats["tables"]["ingest_records"] == 1
    assert stats["tables"]["models"] == 1
    assert stats["total_rows"] == 4
