"""SQLite persistence helpers for API events and model metadata."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class SQLiteStore:
    """Small SQLite wrapper for predictions, ingest events, and model metadata."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _to_int_flag(value: bool | None) -> int | None:
        if value is None:
            return None
        return 1 if value else 0

    @staticmethod
    def _to_json(value: Any | None) -> str | None:
        if value is None:
            return None
        return json.dumps(value)

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        create_tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                request_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL,
                model_version TEXT NOT NULL,
                probabilities_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ingest_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT NOT NULL,
                num_records INTEGER NOT NULL,
                min_window_size INTEGER NOT NULL,
                drift_threshold REAL NOT NULL,
                auto_retrain INTEGER NOT NULL,
                drift_score REAL,
                is_drift INTEGER NOT NULL,
                retrained INTEGER NOT NULL,
                promoted INTEGER,
                champion_version TEXT,
                challenger_version TEXT,
                new_version TEXT,
                message TEXT NOT NULL,
                drift_details_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ingest_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                original_id INTEGER,
                text TEXT NOT NULL,
                label TEXT,
                FOREIGN KEY(batch_id) REFERENCES ingest_batches(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS models (
                version TEXT PRIMARY KEY,
                saved_at TEXT NOT NULL,
                metrics_json TEXT,
                config_json TEXT,
                promoted INTEGER NOT NULL DEFAULT 0,
                notes TEXT
            )
            """,
        ]

        with self._connect() as conn:
            for sql in create_tables_sql:
                conn.execute(sql)

    def log_prediction(
        self,
        *,
        request_text: str,
        prediction: str,
        probability: float | None,
        model_version: str,
        probabilities: dict[str, float] | None,
    ) -> None:
        timestamp = self._utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    timestamp, request_text, prediction, probability,
                    model_version, probabilities_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    request_text,
                    prediction,
                    probability,
                    model_version,
                    self._to_json(probabilities),
                ),
            )

    def log_ingest_batch(
        self,
        *,
        num_records: int,
        min_window_size: int,
        drift_threshold: float,
        auto_retrain: bool,
        drift_score: float | None,
        is_drift: bool,
        retrained: bool,
        promoted: bool | None,
        champion_version: str | None,
        challenger_version: str | None,
        new_version: str | None,
        message: str,
        drift_details: dict[str, Any],
    ) -> int:
        received_at = self._utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO ingest_batches (
                    received_at,
                    num_records,
                    min_window_size,
                    drift_threshold,
                    auto_retrain,
                    drift_score,
                    is_drift,
                    retrained,
                    promoted,
                    champion_version,
                    challenger_version,
                    new_version,
                    message,
                    drift_details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    received_at,
                    num_records,
                    min_window_size,
                    drift_threshold,
                    self._to_int_flag(auto_retrain),
                    drift_score,
                    self._to_int_flag(is_drift),
                    self._to_int_flag(retrained),
                    self._to_int_flag(promoted),
                    champion_version,
                    challenger_version,
                    new_version,
                    message,
                    self._to_json(drift_details),
                ),
            )
            return int(cur.lastrowid)

    def log_ingest_records(self, batch_id: int, records: Iterable[dict[str, Any]]) -> None:
        rows: list[tuple[int, int | None, str, str | None]] = [
            (batch_id, r.get("id"), str(r.get("text", "")), r.get("label")) for r in records
        ]

        if not rows:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO ingest_records (batch_id, original_id, text, label)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def upsert_model(
        self,
        *,
        version: str,
        metrics: dict[str, Any] | None,
        config: dict[str, Any] | None,
        promoted: bool,
        notes: str | None = None,
    ) -> None:
        saved_at = self._utc_now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT version FROM models WHERE version = ?",
                (version,),
            ).fetchone()

            payload = (
                saved_at,
                self._to_json(metrics),
                self._to_json(config),
                self._to_int_flag(promoted),
                notes,
                version,
            )

            if existing:
                conn.execute(
                    """
                    UPDATE models
                    SET saved_at = ?,
                        metrics_json = ?,
                        config_json = ?,
                        promoted = ?,
                        notes = ?
                    WHERE version = ?
                    """,
                    payload,
                )
            else:
                conn.execute(
                    """
                    INSERT INTO models (
                        saved_at,
                        metrics_json,
                        config_json,
                        promoted,
                        notes,
                        version
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )

    def get_stats(self) -> dict[str, Any]:
        """Return row counts for tracked tables and db file metadata."""
        table_names = ["predictions", "ingest_batches", "ingest_records", "models"]
        counts: dict[str, int] = {}

        with self._connect() as conn:
            for table in table_names:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = int(row[0]) if row else 0

        total_rows = sum(counts.values())
        db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "db_path": str(self.db_path),
            "db_exists": self.db_path.exists(),
            "db_size_bytes": db_size_bytes,
            "tables": counts,
            "total_rows": total_rows,
        }
