"""MessageLog: persist raw group chat messages to SQLite."""

from __future__ import annotations

import time
from typing import Any

import aiosqlite
from loguru import logger

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS group_messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id     TEXT    NOT NULL,
    role         TEXT    NOT NULL,
    speaker      TEXT,
    content_text TEXT,
    content_json TEXT,
    message_id   INTEGER,
    created_at   REAL    NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_gm_group_time ON group_messages(group_id, created_at)",
]

_INSERT = """
INSERT INTO group_messages
    (group_id, role, speaker, content_text, content_json, message_id, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_QUERY_FOR_COMPACT = """
SELECT role, speaker, content_text, content_json, message_id, created_at
FROM group_messages
WHERE group_id = ? AND created_at <= ?
ORDER BY created_at
"""


class MessageLog:
    def __init__(self, db_path: str = "storage/messages.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Connect to SQLite and create table/index if needed."""
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_TABLE)
        for idx in _CREATE_INDEXES:
            await self._db.execute(idx)
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def record(
        self,
        *,
        group_id: str,
        role: str,
        speaker: str | None,
        content_text: str | None,
        content_json: str | None,
        message_id: int | None = None,
    ) -> None:
        """Insert a message row. Errors are logged and swallowed."""
        if not self._db:
            logger.warning("message_log not initialized, skipping record")
            return
        try:
            await self._db.execute(
                _INSERT,
                (group_id, role, speaker, content_text, content_json,
                 message_id, time.time()),
            )
            await self._db.commit()
        except Exception:
            logger.exception("message_log record failed")

    async def query_for_compact(
        self,
        group_id: str,
        *,
        before: float,
    ) -> list[dict[str, Any]]:
        """Return messages for compaction: all rows up to *before* timestamp."""
        if not self._db:
            return []
        cursor = await self._db.execute(_QUERY_FOR_COMPACT, (group_id, before))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
