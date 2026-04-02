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
