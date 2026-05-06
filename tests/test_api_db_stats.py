import asyncio
from pathlib import Path

from app.db.sqlite_store import SQLiteStore


def test_db_stats_endpoint_returns_counts(tmp_path: Path):
    db_path = tmp_path / "driftguard.db"
    store = SQLiteStore(db_path)
    store.init_db()
    store.log_prediction(
        request_text="hello",
        prediction="positive",
        probability=None,
        model_version="model_v1",
        probabilities=None,
    )

    # Override global db_store used by endpoint for this test.
    import app.api.server as server_module

    server_module.db_store = store

    body = asyncio.run(server_module.db_stats())

    assert body["tables"]["predictions"] == 1
    assert "total_rows" in body
