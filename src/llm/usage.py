"""LLM usage tracking: record API calls to SQLite, query summaries."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import aiosqlite
from loguru import logger

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    call_type       TEXT    NOT NULL,
    user_id         TEXT,
    group_id        TEXT,
    model           TEXT    NOT NULL,
    input_tokens    INTEGER NOT NULL,
    cache_read_tokens  INTEGER NOT NULL,
    cache_create_tokens INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    tool_rounds     INTEGER NOT NULL,
    elapsed_s       REAL    NOT NULL,
    error           TEXT
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_llm_calls_ts ON llm_calls (ts)",
    "CREATE INDEX IF NOT EXISTS idx_llm_calls_user ON llm_calls (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_llm_calls_group ON llm_calls (group_id)",
    "CREATE INDEX IF NOT EXISTS idx_llm_calls_type ON llm_calls (call_type)",
]

_INSERT = """
INSERT INTO llm_calls
    (ts, call_type, user_id, group_id, model,
     input_tokens, cache_read_tokens, cache_create_tokens, output_tokens,
     tool_rounds, elapsed_s, error)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class UsageTracker:
    def __init__(self, db_path: str = "storage/usage.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_TABLE)
        for idx in _CREATE_INDEXES:
            await self._db.execute(idx)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def record(
        self,
        *,
        call_type: str,
        user_id: str | None,
        group_id: str | None,
        model: str,
        input_tokens: int,
        cache_read_tokens: int,
        cache_create_tokens: int,
        output_tokens: int,
        tool_rounds: int,
        elapsed_s: float,
        error: str | None = None,
    ) -> None:
        if not self._db:
            logger.warning("usage tracker not initialized, skipping record")
            return
        ts = datetime.now(UTC).isoformat()
        try:
            await self._db.execute(
                _INSERT,
                (ts, call_type, user_id, group_id, model,
                 input_tokens, cache_read_tokens, cache_create_tokens, output_tokens,
                 tool_rounds, elapsed_s, error),
            )
            await self._db.commit()
            logger.info(
                "usage | type={} user={} group={} in={} out={} cache_r={} cache_w={} rounds={} {:.1f}s{}",
                call_type, user_id, group_id,
                input_tokens, output_tokens, cache_read_tokens, cache_create_tokens,
                tool_rounds, elapsed_s,
                f" error={error}" if error else "",
            )
        except Exception:
            logger.exception("usage record failed")

    async def query_raw(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Run arbitrary SQL and return rows as dicts. For internal use and CLI."""
        if not self._db:
            return []
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def summary_today(self) -> dict[str, Any]:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return await self._summary_for_period(f"{today}%")

    async def summary_month(self, month: str | None = None) -> dict[str, Any]:
        if month is None:
            month = datetime.now(UTC).strftime("%Y-%m")
        return await self._summary_for_period(f"{month}%")

    async def _summary_for_period(self, ts_like: str) -> dict[str, Any]:
        rows = await self.query_raw(
            """
            SELECT
                COUNT(*)        AS total_calls,
                COALESCE(SUM(input_tokens + cache_read_tokens + cache_create_tokens), 0) AS total_input_tokens,
                COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
                COALESCE(SUM(CASE WHEN call_type='chat' THEN 1 ELSE 0 END), 0) AS chat_calls,
                COALESCE(SUM(CASE WHEN call_type='proactive' THEN 1 ELSE 0 END), 0) AS proactive_calls,
                COALESCE(SUM(CASE WHEN call_type='compact' THEN 1 ELSE 0 END), 0) AS compact_calls,
                COALESCE(SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END), 0) AS error_count,
                COALESCE(AVG(elapsed_s), 0) AS avg_elapsed_s
            FROM llm_calls WHERE ts LIKE ?
            """,
            (ts_like,),
        )
        return rows[0] if rows else {}

    async def top_users(self, days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
        return await self.query_raw(
            """
            SELECT user_id,
                   COUNT(*) AS calls,
                   SUM(input_tokens + cache_read_tokens + cache_create_tokens) AS total_input,
                   SUM(output_tokens) AS total_output
            FROM llm_calls
            WHERE user_id IS NOT NULL
              AND ts >= datetime('now', ?)
            GROUP BY user_id
            ORDER BY total_input + total_output DESC
            LIMIT ?
            """,
            (f"-{days} days", limit),
        )

    async def top_groups(self, days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
        return await self.query_raw(
            """
            SELECT group_id,
                   COUNT(*) AS calls,
                   SUM(input_tokens + cache_read_tokens + cache_create_tokens) AS total_input,
                   SUM(output_tokens) AS total_output
            FROM llm_calls
            WHERE group_id IS NOT NULL
              AND ts >= datetime('now', ?)
            GROUP BY group_id
            ORDER BY total_input + total_output DESC
            LIMIT ?
            """,
            (f"-{days} days", limit),
        )
