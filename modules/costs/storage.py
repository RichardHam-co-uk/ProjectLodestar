"""SQLite persistence for cost tracking data.

Provides durable storage and querying for cost records. The tracker
uses this to persist data across sessions.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cost_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    baseline_cost_usd REAL NOT NULL,
    savings_usd REAL NOT NULL,
    task TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_cost_timestamp ON cost_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_cost_model ON cost_records(model);
"""


class CostStorage:
    """SQLite storage backend for cost records.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open database connection and create schema if needed."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _check_connected(self) -> None:
        """Raise RuntimeError if not connected."""
        if not self._conn:
            raise RuntimeError("Database not connected. Call connect() first.")

    def insert(self, record: Dict[str, Any]) -> None:
        """Insert a single cost record.

        Args:
            record: Dict with model, tokens_in, tokens_out, cost,
                    baseline_cost, savings, task keys.
        """
        self._check_connected()
        self._conn.execute(
            """INSERT INTO cost_records
               (model, tokens_in, tokens_out, cost_usd, baseline_cost_usd,
                savings_usd, task)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                record["model"],
                record["tokens_in"],
                record["tokens_out"],
                record["cost"],
                record["baseline_cost"],
                record["savings"],
                record.get("task", ""),
            ),
        )
        self._conn.commit()

    def query_all(self) -> List[Dict[str, Any]]:
        """Retrieve all cost records.

        Returns:
            List of record dicts ordered by timestamp descending.
        """
        self._check_connected()
        cursor = self._conn.execute(
            "SELECT * FROM cost_records ORDER BY timestamp DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def query_by_model(self, model: str) -> List[Dict[str, Any]]:
        """Retrieve cost records for a specific model.

        Args:
            model: Model alias to filter by.

        Returns:
            List of record dicts ordered by timestamp descending.
        """
        self._check_connected()
        cursor = self._conn.execute(
            "SELECT * FROM cost_records WHERE model = ? ORDER BY timestamp DESC",
            (model,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def query_by_date_range(
        self, start: datetime, end: datetime
    ) -> List[Dict[str, Any]]:
        """Retrieve cost records within a date range.

        Args:
            start: Start of date range (inclusive).
            end: End of date range (inclusive).

        Returns:
            List of record dicts ordered by timestamp descending.
        """
        self._check_connected()
        cursor = self._conn.execute(
            "SELECT * FROM cost_records WHERE timestamp BETWEEN ? AND ? "
            "ORDER BY timestamp DESC",
            (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")),
        )
        return [dict(row) for row in cursor.fetchall()]

    def total_cost(self) -> float:
        """Get total cost from the database.

        Returns:
            Total cost in USD.
        """
        self._check_connected()
        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_records"
        )
        return cursor.fetchone()[0]

    def total_savings(self) -> float:
        """Get total savings from the database.

        Returns:
            Total savings in USD.
        """
        self._check_connected()
        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(savings_usd), 0) FROM cost_records"
        )
        return cursor.fetchone()[0]

    def cleanup(self, retention_days: int = 90) -> int:
        """Delete records older than the retention period.

        Args:
            retention_days: Number of days to retain records.

        Returns:
            Number of records deleted.
        """
        self._check_connected()
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM cost_records WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info("Cleaned up %d records older than %d days", deleted, retention_days)
        return deleted

    def record_count(self) -> int:
        """Get total number of records in the database.

        Returns:
            Record count.
        """
        self._check_connected()
        cursor = self._conn.execute("SELECT COUNT(*) FROM cost_records")
        return cursor.fetchone()[0]
