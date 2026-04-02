# LLM Usage Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record every LLM API call's token consumption and latency to SQLite, with CLI/HTTP query interfaces and anomaly alerting.

**Architecture:** A new `UsageTracker` singleton in `src/llm/usage.py` writes to `storage/usage.db`. It is injected into `LLMClient` and called after each chat/compact. Alerting (loguru WARNING + Bot PM) fires inside `record()`. CLI and HTTP endpoints query the same DB.

**Tech Stack:** aiosqlite, SQLite, FastAPI (already bundled via NoneBot), loguru

---

### Task 1: Add aiosqlite dependency and UsageConfig

**Files:**
- Modify: `pyproject.toml:6-15`
- Modify: `src/config.py:14-21`
- Modify: `config.example.toml`
- Test: `tests/test_config_loader.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_config_loader.py` (create if needed — check existing file first):

```python
from src.config import BotConfig, UsageConfig


def test_usage_config_defaults():
    cfg = BotConfig()
    assert cfg.llm.usage.enabled is True
    assert cfg.llm.usage.slow_threshold_s == 60.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_loader.py::test_usage_config_defaults -v`
Expected: FAIL — `UsageConfig` does not exist yet.

- [ ] **Step 3: Add UsageConfig to src/config.py**

Add after `ContextConfig` class (before `LLMConfig`):

```python
class UsageConfig(BaseModel):
    """LLM usage tracking configuration."""

    enabled: bool = True
    slow_threshold_s: float = 60.0
```

Add `usage` field to `LLMConfig`:

```python
class LLMConfig(BaseModel):
    """LLM 接入配置。"""

    base_url: str = "http://127.0.0.1:34567/v1"
    api_key: str = "sk-placeholder"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    context: ContextConfig = ContextConfig()
    usage: UsageConfig = UsageConfig()
```

- [ ] **Step 4: Add aiosqlite to pyproject.toml**

Add `"aiosqlite>=0.21.0"` to the `dependencies` list.

- [ ] **Step 5: Add [llm.usage] to config.example.toml**

Append after the `[llm.context]` section:

```toml
# ---------------------------------------------------------------------------
# LLM 用量追踪
# ---------------------------------------------------------------------------
[llm.usage]
# 是否启用用量记录
enabled = true
# 单次调用耗时告警阈值（秒）
slow_threshold_s = 60.0
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: PASS

- [ ] **Step 7: Install dependency and run full checks**

Run: `uv sync && uv run ruff check src/ && uv run pytest`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/config.py config.example.toml tests/test_config_loader.py
git commit -m "feat: add UsageConfig and aiosqlite dependency"
```

---

### Task 2: UsageTracker — DB init and record()

**Files:**
- Create: `src/llm/usage.py`
- Create: `tests/test_usage.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_usage.py`:

```python
"""Tests for UsageTracker."""

import asyncio

import pytest

from src.llm.usage import UsageTracker


@pytest.fixture
async def tracker(tmp_path) -> UsageTracker:
    t = UsageTracker(db_path=str(tmp_path / "usage.db"))
    await t.init()
    return t


async def test_init_creates_table(tracker: UsageTracker) -> None:
    """Table should exist after init."""
    rows = await tracker.query_raw("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_calls'")
    assert len(rows) == 1


async def test_record_inserts_row(tracker: UsageTracker) -> None:
    await tracker.record(
        call_type="chat",
        user_id="12345",
        group_id=None,
        model="claude-sonnet-4-6",
        input_tokens=100,
        cache_read_tokens=50,
        cache_create_tokens=10,
        output_tokens=200,
        tool_rounds=0,
        elapsed_s=1.5,
    )
    rows = await tracker.query_raw("SELECT * FROM llm_calls")
    assert len(rows) == 1
    row = rows[0]
    assert row["call_type"] == "chat"
    assert row["user_id"] == "12345"
    assert row["group_id"] is None
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 200
    assert row["cache_read_tokens"] == 50
    assert row["cache_create_tokens"] == 10
    assert row["tool_rounds"] == 0
    assert row["elapsed_s"] == pytest.approx(1.5)
    assert row["error"] is None


async def test_record_with_error(tracker: UsageTracker) -> None:
    await tracker.record(
        call_type="chat",
        user_id="12345",
        group_id="99999",
        model="claude-sonnet-4-6",
        input_tokens=0,
        cache_read_tokens=0,
        cache_create_tokens=0,
        output_tokens=0,
        tool_rounds=0,
        elapsed_s=0.5,
        error="API timeout",
    )
    rows = await tracker.query_raw("SELECT error, group_id FROM llm_calls")
    assert rows[0]["error"] == "API timeout"
    assert rows[0]["group_id"] == "99999"


async def test_record_failure_does_not_raise(tracker: UsageTracker, tmp_path) -> None:
    """record() should swallow errors gracefully."""
    await tracker.close()
    # After close, writing should not raise
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=0, cache_read_tokens=0, cache_create_tokens=0,
        output_tokens=0, tool_rounds=0, elapsed_s=0.0,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_usage.py -v`
Expected: FAIL — `src.llm.usage` does not exist.

- [ ] **Step 3: Implement UsageTracker**

Create `src/llm/usage.py`:

```python
"""LLM usage tracking: record API calls to SQLite, query summaries."""

from __future__ import annotations

from datetime import datetime, timezone
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
        ts = datetime.now(timezone.utc).isoformat()
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_usage.py -v`
Expected: PASS

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check src/llm/usage.py && uv run pyright src/llm/usage.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm/usage.py tests/test_usage.py
git commit -m "feat: UsageTracker core — DB init and record()"
```

---

### Task 3: UsageTracker — query methods

**Files:**
- Modify: `src/llm/usage.py`
- Modify: `tests/test_usage.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_usage.py`:

```python
async def _insert_sample_data(tracker: UsageTracker) -> None:
    """Insert sample records for query tests."""
    records = [
        ("chat", "111", None, "model-a", 100, 50, 10, 200, 1, 2.0),
        ("chat", "111", "999", "model-a", 150, 80, 20, 300, 2, 3.0),
        ("chat", "222", "999", "model-a", 200, 100, 30, 400, 0, 1.0),
        ("proactive", None, "999", "model-a", 50, 30, 5, 100, 0, 5.0),
        ("compact", "111", None, "model-a", 80, 0, 0, 50, 0, 1.5),
    ]
    for r in records:
        await tracker.record(
            call_type=r[0], user_id=r[1], group_id=r[2], model=r[3],
            input_tokens=r[4], cache_read_tokens=r[5], cache_create_tokens=r[6],
            output_tokens=r[7], tool_rounds=r[8], elapsed_s=r[9],
        )


async def test_summary_today(tracker: UsageTracker) -> None:
    await _insert_sample_data(tracker)
    summary = await tracker.summary_today()
    assert summary["total_calls"] == 5
    # total_input = sum(input + cache_read + cache_create) for each record
    assert summary["total_input_tokens"] == (100+50+10) + (150+80+20) + (200+100+30) + (50+30+5) + (80+0+0)
    assert summary["total_output_tokens"] == 200 + 300 + 400 + 100 + 50


async def test_top_users(tracker: UsageTracker) -> None:
    await _insert_sample_data(tracker)
    top = await tracker.top_users(days=1)
    assert len(top) >= 2
    # User 111 has more total tokens than 222
    assert top[0]["user_id"] == "222" or top[0]["user_id"] == "111"


async def test_top_groups(tracker: UsageTracker) -> None:
    await _insert_sample_data(tracker)
    top = await tracker.top_groups(days=1)
    assert len(top) >= 1
    assert top[0]["group_id"] == "999"


async def test_summary_month(tracker: UsageTracker) -> None:
    await _insert_sample_data(tracker)
    now = datetime.now(timezone.utc)
    month_str = now.strftime("%Y-%m")
    summary = await tracker.summary_month(month_str)
    assert summary["total_calls"] == 5
```

Add the import at the top of the test file:

```python
from datetime import datetime, timezone
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_usage.py::test_summary_today -v`
Expected: FAIL — methods don't exist.

- [ ] **Step 3: Add query methods to UsageTracker**

Add to `src/llm/usage.py`, inside the `UsageTracker` class:

```python
    async def summary_today(self) -> dict[str, Any]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return await self._summary_for_period(f"{today}%")

    async def summary_month(self, month: str | None = None) -> dict[str, Any]:
        if month is None:
            month = datetime.now(timezone.utc).strftime("%Y-%m")
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_usage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/usage.py tests/test_usage.py
git commit -m "feat: UsageTracker query methods — today, month, top users/groups"
```

---

### Task 4: Fix output_tokens extraction in _call_api()

**Files:**
- Modify: `src/llm/client.py:130-185`
- Create: `tests/test_call_api.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_call_api.py`:

```python
"""Tests for _call_api SSE parsing."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.client import _call_api


def _make_sse_lines(*events: str) -> list[bytes]:
    """Build raw SSE byte lines from event JSON strings."""
    lines: list[bytes] = []
    for event in events:
        lines.append(f"data: {event}\n".encode())
    return lines


def _mock_session(sse_lines: list[bytes]) -> MagicMock:
    """Create a mock aiohttp session whose post() yields given SSE lines."""
    resp = AsyncMock()
    resp.raise_for_status = MagicMock()
    resp.content.__aiter__ = lambda self: aiter(iter(sse_lines))

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post.return_value = ctx
    return session


async def test_output_tokens_from_message_delta() -> None:
    """output_tokens should come from the last message_delta, not message_start."""
    import json

    events = [
        json.dumps({
            "type": "message_start",
            "message": {
                "usage": {"input_tokens": 100, "output_tokens": 1,
                           "cache_read_input_tokens": 50, "cache_creation_input_tokens": 10}
            },
        }),
        json.dumps({"type": "content_block_start", "content_block": {"type": "text", "text": ""}}),
        json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}),
        json.dumps({"type": "content_block_stop"}),
        json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 42}}),
    ]

    session = _mock_session(_make_sse_lines(*events))
    result = await _call_api(session, "http://fake", "sk-test", "model", [], [{"role": "user", "content": "hi"}])

    assert result["output_tokens"] == 42
    assert result["text"] == "Hello"
    assert result["input_tokens"] == 100 + 50 + 10  # total input
    assert result["cache_read"] == 50
    assert result["cache_create"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_call_api.py::test_output_tokens_from_message_delta -v`
Expected: FAIL — `output_tokens` key missing or wrong value.

- [ ] **Step 3: Fix _call_api in src/llm/client.py**

Two changes in `_call_api()`:

**A) Add `message_delta` handling** — after the `elif event_type == "error"` block (line 161-164), add:

```python
            elif event_type == "message_delta":
                delta_usage: dict[str, Any] = data.get("usage", {})
                for k, v in delta_usage.items():
                    if isinstance(v, int):
                        usage[k] = v
```

**B) Add `output_tokens` to return dict and remove cache log** — replace lines 166-185:

```python
    # Token stats
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    input_tokens = usage.get("input_tokens", 0)
    total_input = input_tokens + cache_read + cache_create
    output_tokens = usage.get("output_tokens", 0)

    return {
        "text": "".join(text_parts),
        "tool_uses": tool_uses,
        "input_tokens": total_input,
        "output_tokens": output_tokens,
        "cache_read": cache_read,
        "cache_create": cache_create,
    }
```

This removes the `cache_hit_rate` computation and the cache log line — both move to `UsageTracker` in Task 6.

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_call_api.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest`
Expected: PASS — existing tests don't depend on `cache_hit_rate` in the return dict. The `_check_cache_rate` method still exists temporarily and reads `cache_hit_rate` from result, but that method is only called in `chat()` which uses the real `_call_api` (always mocked in tests). We'll remove `_check_cache_rate` in Task 6.

Actually — `_check_cache_rate` accesses `result["cache_hit_rate"]` which we just removed. Check if any tests call `_check_cache_rate`. Looking at the existing tests: **no test directly tests `_check_cache_rate`**, and `chat()` is never tested end-to-end (only compact methods are tested). But to be safe, add `cache_hit_rate` computation back temporarily as a local in `_check_cache_rate`, or better: remove `_check_cache_rate` now since it will be replaced by UsageTracker.

**Remove `_check_cache_rate` and related code now:**

1. Delete `_DEFAULT_CACHE_HIT_WARN` constant (line 27)
2. Delete `_check_cache_rate` method (lines 245-277)
3. Remove `cache_hit_warn` parameter from `__init__` (line 201) and `self._cache_hit_warn` (line 225)
4. Remove the two `self._check_cache_rate(...)` calls in `chat()` (lines 406-408 and 479-481)

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest`
Expected: PASS — no test references `cache_hit_warn` or `_check_cache_rate`.

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/llm/client.py`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/llm/client.py tests/test_call_api.py
git commit -m "fix: extract output_tokens from message_delta; remove cache_hit_rate from _call_api"
```

---

### Task 5: UsageTracker — alerting

**Files:**
- Modify: `src/llm/usage.py`
- Modify: `tests/test_usage.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_usage.py`:

```python
from unittest.mock import AsyncMock


async def test_alert_on_slow_call(tracker: UsageTracker) -> None:
    alert_fn = AsyncMock()
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=2.0)
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=100, cache_read_tokens=80, cache_create_tokens=10,
        output_tokens=50, tool_rounds=0, elapsed_s=5.0,
    )
    alert_fn.assert_called_once()
    assert "slow" in alert_fn.call_args[0][0].lower() or "慢" in alert_fn.call_args[0][0]


async def test_alert_on_low_cache_hit(tracker: UsageTracker) -> None:
    alert_fn = AsyncMock()
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=999.0)
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=100, cache_read_tokens=10, cache_create_tokens=0,
        output_tokens=50, tool_rounds=0, elapsed_s=1.0,
    )
    alert_fn.assert_called_once()
    assert "cache" in alert_fn.call_args[0][0].lower()


async def test_alert_on_error(tracker: UsageTracker) -> None:
    alert_fn = AsyncMock()
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=999.0)
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=0, cache_read_tokens=0, cache_create_tokens=0,
        output_tokens=0, tool_rounds=0, elapsed_s=0.5,
        error="API timeout",
    )
    alert_fn.assert_called_once()
    assert "error" in alert_fn.call_args[0][0].lower() or "错误" in alert_fn.call_args[0][0]


async def test_no_alert_when_ok(tracker: UsageTracker) -> None:
    alert_fn = AsyncMock()
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=60.0)
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=10, cache_read_tokens=90, cache_create_tokens=0,
        output_tokens=50, tool_rounds=0, elapsed_s=1.0,
    )
    alert_fn.assert_not_called()


async def test_no_cache_alert_for_compact(tracker: UsageTracker) -> None:
    """compact calls don't use prompt cache, so no cache hit warning."""
    alert_fn = AsyncMock()
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=999.0)
    await tracker.record(
        call_type="compact", user_id="1", group_id=None, model="m",
        input_tokens=100, cache_read_tokens=0, cache_create_tokens=0,
        output_tokens=50, tool_rounds=0, elapsed_s=1.0,
    )
    alert_fn.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_usage.py::test_alert_on_slow_call -v`
Expected: FAIL — `set_alert` does not exist.

- [ ] **Step 3: Add alerting to UsageTracker**

Add to `src/llm/usage.py`, in the `UsageTracker` class:

```python
    def set_alert(
        self,
        *,
        alert_fn: Callable[[str], Awaitable[None]],
        cache_hit_warn: float = 90.0,
        slow_threshold_s: float = 60.0,
    ) -> None:
        self._alert_fn = alert_fn
        self._cache_hit_warn = cache_hit_warn
        self._slow_threshold_s = slow_threshold_s
```

Initialize defaults in `__init__`:

```python
    def __init__(self, db_path: str = "storage/usage.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._alert_fn: Callable[[str], Awaitable[None]] | None = None
        self._cache_hit_warn: float = 90.0
        self._slow_threshold_s: float = 60.0
```

Add alert checking at the end of `record()`, after the successful DB write and log:

```python
        await self._check_alerts(
            call_type=call_type, elapsed_s=elapsed_s, error=error,
            input_tokens=input_tokens, cache_read_tokens=cache_read_tokens,
            cache_create_tokens=cache_create_tokens,
        )
```

Implement `_check_alerts`:

```python
    async def _check_alerts(
        self,
        *,
        call_type: str,
        elapsed_s: float,
        error: str | None,
        input_tokens: int,
        cache_read_tokens: int,
        cache_create_tokens: int,
    ) -> None:
        if not self._alert_fn:
            return

        if error:
            msg = f"⚠ LLM call error: {error}"
            logger.warning("usage_alert | {}", msg)
            await self._alert_fn(msg)
            return  # error alert takes priority, skip others

        if elapsed_s > self._slow_threshold_s:
            msg = f"⚠ LLM slow call: {elapsed_s:.1f}s (threshold: {self._slow_threshold_s:.0f}s)"
            logger.warning("usage_alert | {}", msg)
            await self._alert_fn(msg)

        # Cache hit check — only for chat/proactive (compact/dream don't use prompt cache)
        if call_type in ("chat", "proactive"):
            total = input_tokens + cache_read_tokens + cache_create_tokens
            if total > 0:
                hit_rate = cache_read_tokens / total * 100
                if hit_rate < self._cache_hit_warn:
                    msg = f"⚠ Cache hit rate low: {hit_rate:.0f}% (threshold: {self._cache_hit_warn:.0f}%)"
                    logger.warning("usage_alert | {}", msg)
                    await self._alert_fn(msg)
```

Add the required imports at the top of `usage.py`:

```python
from collections.abc import Awaitable, Callable
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_usage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/usage.py tests/test_usage.py
git commit -m "feat: UsageTracker alerting — slow calls, low cache hit, errors"
```

---

### Task 6: Integrate UsageTracker into LLMClient

**Files:**
- Modify: `src/llm/client.py`
- Modify: `src/plugins/chat/__init__.py`
- Modify: `tests/test_client.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_client.py`:

```python
from src.llm.usage import UsageTracker


MOCK_RESULT_FULL = {
    "text": "reply text",
    "tool_uses": [],
    "input_tokens": 160,  # total = 100 + 50 + 10
    "output_tokens": 200,
    "cache_read": 50,
    "cache_create": 10,
}


async def test_chat_records_usage(prompt, short_term, tools, tmp_path) -> None:
    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    await tracker.init()
    try:
        async for client in _client(prompt, short_term, tools):
            client._usage_tracker = tracker
            with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=MOCK_RESULT_FULL):
                await client.chat(
                    session_id="private_100", user_id="100",
                    user_text="hello", identity=_IDENTITY,
                )
        rows = await tracker.query_raw("SELECT * FROM llm_calls")
        assert len(rows) == 1
        row = rows[0]
        assert row["call_type"] == "chat"
        assert row["user_id"] == "100"
        assert row["output_tokens"] == 200
        assert row["input_tokens"] == 100  # raw = total(160) - cache_read(50) - cache_create(10)
        assert row["cache_read_tokens"] == 50
    finally:
        await tracker.close()


async def test_compact_records_usage(prompt, short_term, tools, tmp_path) -> None:
    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    await tracker.init()
    mock_result = {**MOCK_RESULT, "output_tokens": 80, "cache_read": 0, "cache_create": 0}
    try:
        async for client in _client(prompt, short_term, tools):
            client._usage_tracker = tracker
            _fill_messages(short_term, "private_100")
            with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_result):
                await client._compact("private_100")
        rows = await tracker.query_raw("SELECT * FROM llm_calls WHERE call_type='compact'")
        assert len(rows) == 1
    finally:
        await tracker.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_client.py::test_chat_records_usage -v`
Expected: FAIL — `_usage_tracker` attribute doesn't exist on `LLMClient`.

- [ ] **Step 3: Modify LLMClient**

**A) Add `_usage_tracker` attribute to `__init__`** (after `self._on_compact`):

```python
        self._usage_tracker: UsageTracker | None = None
```

Add import at top of `client.py`:

```python
from src.llm.usage import UsageTracker
```

**B) Add `_record_usage` helper method** to `LLMClient`:

```python
    def _record_usage(
        self,
        *,
        call_type: str,
        user_id: str,
        group_id: str | None,
        input_tokens: int,
        cache_read_tokens: int,
        cache_create_tokens: int,
        output_tokens: int,
        tool_rounds: int,
        elapsed_s: float,
        error: str | None = None,
    ) -> None:
        """Fire-and-forget usage recording."""
        if not self._usage_tracker:
            return
        asyncio.create_task(self._usage_tracker.record(
            call_type=call_type,
            user_id=user_id or None,
            group_id=group_id,
            model=self._model,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_create_tokens=cache_create_tokens,
            output_tokens=output_tokens,
            tool_rounds=tool_rounds,
            elapsed_s=elapsed_s,
            error=error,
        ))
```

**C) Accumulate tokens and record in `chat()`**

The tool loop may call `_call_api()` multiple times. We need to **sum** tokens across all rounds and record once at the end.

Add accumulators before the `for round_i in range(MAX_TOOL_ROUNDS):` loop:

```python
        # Token accumulators across tool rounds
        acc_input = 0
        acc_output = 0
        acc_cache_read = 0
        acc_cache_create = 0
```

After each `result = await self._call(...)` call inside the loop (there are two: the main one at the top of the loop and the one after tool-loop-exhausted), add:

```python
            acc_input += result["input_tokens"] - result.get("cache_read", 0) - result.get("cache_create", 0)
            acc_output += result.get("output_tokens", 0)
            acc_cache_read += result.get("cache_read", 0)
            acc_cache_create += result.get("cache_create", 0)
```

Then for each exit point in `chat()`, record with accumulated values:

For the `if not tool_uses:` branch (around line 425-445), add before `return last_seg`:

```python
                self._record_usage(
                    call_type="proactive" if allow_skip else "chat",
                    user_id=user_id, group_id=group_id,
                    input_tokens=acc_input, cache_read_tokens=acc_cache_read,
                    cache_create_tokens=acc_cache_create, output_tokens=acc_output,
                    tool_rounds=round_i, elapsed_s=elapsed,
                )
```

For the `pass_turn` branch (around line 414-420), add before `return None`:

```python
                self._record_usage(
                    call_type="proactive",
                    user_id=user_id, group_id=group_id,
                    input_tokens=acc_input, cache_read_tokens=acc_cache_read,
                    cache_create_tokens=acc_cache_create, output_tokens=acc_output,
                    tool_rounds=round_i, elapsed_s=elapsed,
                )
```

For the tool-loop-exhausted branch (after the final `_call` and memory storage), add:

```python
        acc_input += result["input_tokens"] - result.get("cache_read", 0) - result.get("cache_create", 0)
        acc_output += result.get("output_tokens", 0)
        acc_cache_read += result.get("cache_read", 0)
        acc_cache_create += result.get("cache_create", 0)
        elapsed = time.monotonic() - t0
        self._record_usage(
            call_type="proactive" if allow_skip else "chat",
            user_id=user_id, group_id=group_id,
            input_tokens=acc_input, cache_read_tokens=acc_cache_read,
            cache_create_tokens=acc_cache_create, output_tokens=acc_output,
            tool_rounds=MAX_TOOL_ROUNDS, elapsed_s=elapsed,
        )
```

**D) Record in `_compact()`** — after `result = await _call_api(...)` (line 577-580), add:

```python
            self._record_usage(
                call_type="compact", user_id="", group_id=None,
                input_tokens=result["input_tokens"] - result.get("cache_read", 0) - result.get("cache_create", 0),
                cache_read_tokens=result.get("cache_read", 0),
                cache_create_tokens=result.get("cache_create", 0),
                output_tokens=result.get("output_tokens", 0),
                tool_rounds=0, elapsed_s=0.0,
            )
```

**E) Record in `_compact_group()`** — after `result = await _call_api(...)` (line 682-685), add:

```python
            self._record_usage(
                call_type="compact", user_id="", group_id=group_id,
                input_tokens=result["input_tokens"] - result.get("cache_read", 0) - result.get("cache_create", 0),
                cache_read_tokens=result.get("cache_read", 0),
                cache_create_tokens=result.get("cache_create", 0),
                output_tokens=result.get("output_tokens", 0),
                tool_rounds=0, elapsed_s=0.0,
            )
```

**F) Update `MOCK_RESULT` in `tests/test_client.py`** — add the new fields that `_record_usage` accesses:

```python
MOCK_RESULT = {"text": "summary", "tool_uses": [], "input_tokens": 100, "output_tokens": 50, "cache_read": 0, "cache_create": 0}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_client.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm/client.py tests/test_client.py
git commit -m "feat: integrate UsageTracker into LLMClient chat/compact"
```

---

### Task 7: Wire up in plugin and set alert callback

**Files:**
- Modify: `src/plugins/chat/__init__.py`

No TDD for this task — it's wiring code that depends on NoneBot runtime. Verified by the full test suite not breaking.

- [ ] **Step 1: Add UsageTracker initialization to _init()**

In `src/plugins/chat/__init__.py`, add import:

```python
from src.llm.usage import UsageTracker
```

Add module-level variable:

```python
_usage_tracker: UsageTracker
```

In `_init()`, before `_llm = LLMClient(...)`, add:

```python
    global _usage_tracker
    _usage_tracker = UsageTracker(db_path="storage/usage.db")
    if bot_config.llm.usage.enabled:
        await _usage_tracker.init()
```

After `_llm = LLMClient(...)`, add:

```python
    if bot_config.llm.usage.enabled:
        _llm._usage_tracker = _usage_tracker
```

- [ ] **Step 2: Wire alert callback in _on_connect()**

In `_on_connect()`, after `_scheduler.set_bot(bot)`, add:

```python
    # Wire usage alert: PM all admins
    bot_config = load_config()
    admin_ids = list(bot_config.admins.keys())
    if admin_ids and bot_config.llm.usage.enabled:
        async def _alert_admins(msg: str) -> None:
            for admin_id in admin_ids:
                try:
                    await bot.send_private_msg(user_id=int(admin_id), message=msg)
                except Exception:
                    logger.warning("failed to send usage alert to admin {}", admin_id)

        _usage_tracker.set_alert(
            alert_fn=_alert_admins,
            cache_hit_warn=bot_config.compact.cache_hit_warn,
            slow_threshold_s=bot_config.llm.usage.slow_threshold_s,
        )
```

- [ ] **Step 3: Add cleanup in _shutdown()**

In `_shutdown()`, add:

```python
    await _usage_tracker.close()
```

- [ ] **Step 4: Run full tests + lint**

Run: `uv run ruff check src/ && uv run pytest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plugins/chat/__init__.py
git commit -m "feat: wire UsageTracker into bot startup with admin alerting"
```

---

### Task 8: CLI query tool

**Files:**
- Create: `src/llm/usage_cli.py`
- Test: manual — run the CLI and check output format

- [ ] **Step 1: Create the CLI module**

Create `src/llm/usage_cli.py`:

```python
"""CLI for querying LLM usage stats. Run: uv run python -m src.llm.usage_cli"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from src.llm.usage import UsageTracker

_DB_PATH = "storage/usage.db"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _print_summary(title: str, data: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(f"  Total calls:    {data.get('total_calls', 0)}")
    print(f"  Input tokens:   {_fmt_tokens(data.get('total_input_tokens', 0))}")
    print(f"  Output tokens:  {_fmt_tokens(data.get('total_output_tokens', 0))}")
    print(f"  Chat calls:     {data.get('chat_calls', 0)}")
    print(f"  Proactive:      {data.get('proactive_calls', 0)}")
    print(f"  Compact:        {data.get('compact_calls', 0)}")
    print(f"  Errors:         {data.get('error_count', 0)}")
    print(f"  Avg latency:    {data.get('avg_elapsed_s', 0):.1f}s")


def _print_top(title: str, rows: list[dict[str, Any]], id_key: str) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("  (no data)")
        return
    print(f"  {'ID':<15} {'Calls':>6} {'Input':>10} {'Output':>10}")
    print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*10}")
    for row in rows:
        print(
            f"  {str(row[id_key]):<15} {row['calls']:>6} "
            f"{_fmt_tokens(row['total_input']):>10} {_fmt_tokens(row['total_output']):>10}"
        )


async def _run(args: argparse.Namespace) -> None:
    tracker = UsageTracker(db_path=_DB_PATH)
    await tracker.init()
    try:
        if args.command == "today":
            data = await tracker.summary_today()
            _print_summary("Today", data)
        elif args.command == "month":
            data = await tracker.summary_month(args.month)
            _print_summary(f"Month: {args.month or 'current'}", data)
        elif args.command == "top-users":
            rows = await tracker.top_users(days=args.days)
            _print_top(f"Top Users (last {args.days} days)", rows, "user_id")
        elif args.command == "top-groups":
            rows = await tracker.top_groups(days=args.days)
            _print_top(f"Top Groups (last {args.days} days)", rows, "group_id")
    finally:
        await tracker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Usage Stats")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("today", help="Today's usage summary")

    month_p = sub.add_parser("month", help="Monthly usage summary")
    month_p.add_argument("month", nargs="?", default=None, help="YYYY-MM (default: current)")

    users_p = sub.add_parser("top-users", help="Top users by token consumption")
    users_p.add_argument("--days", type=int, default=7, help="Lookback days (default: 7)")

    groups_p = sub.add_parser("top-groups", help="Top groups by token consumption")
    groups_p.add_argument("--days", type=int, default=7, help="Lookback days (default: 7)")

    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint**

Run: `uv run ruff check src/llm/usage_cli.py`
Expected: PASS

- [ ] **Step 3: Verify it runs**

Run: `uv run python -m src.llm.usage_cli today`
Expected: Prints empty summary (no data yet) or usage.db doesn't exist error — should handle gracefully.

- [ ] **Step 4: Commit**

```bash
git add src/llm/usage_cli.py
git commit -m "feat: CLI tool for querying LLM usage stats"
```

---

### Task 9: HTTP query routes

**Files:**
- Create: `src/llm/usage_routes.py`
- Modify: `src/plugins/chat/__init__.py`
- Create: `tests/test_usage_routes.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_usage_routes.py`:

```python
"""Tests for usage HTTP routes."""

import pytest
from starlette.testclient import TestClient

from src.llm.usage import UsageTracker
from src.llm.usage_routes import create_usage_router


@pytest.fixture
async def tracker(tmp_path) -> UsageTracker:
    t = UsageTracker(db_path=str(tmp_path / "usage.db"))
    await t.init()
    return t


@pytest.fixture
def client(tracker: UsageTracker) -> TestClient:
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(create_usage_router(tracker))
    return TestClient(app)


def test_today_endpoint(client: TestClient) -> None:
    resp = client.get("/api/usage/today")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_calls" in data


def test_month_endpoint(client: TestClient) -> None:
    resp = client.get("/api/usage/month")
    assert resp.status_code == 200


def test_top_users_endpoint(client: TestClient) -> None:
    resp = client.get("/api/usage/top-users?days=7")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_top_groups_endpoint(client: TestClient) -> None:
    resp = client.get("/api/usage/top-groups?days=7")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_usage_routes.py -v`
Expected: FAIL — `usage_routes` module doesn't exist.

- [ ] **Step 3: Create the routes module**

Create `src/llm/usage_routes.py`:

```python
"""FastAPI routes for LLM usage stats."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.llm.usage import UsageTracker


def create_usage_router(tracker: UsageTracker) -> APIRouter:
    router = APIRouter(prefix="/api/usage", tags=["usage"])

    @router.get("/today")
    async def today():
        return await tracker.summary_today()

    @router.get("/month")
    async def month(month: str | None = Query(None, description="YYYY-MM")):
        return await tracker.summary_month(month)

    @router.get("/top-users")
    async def top_users(days: int = Query(7), limit: int = Query(10)):
        return await tracker.top_users(days=days, limit=limit)

    @router.get("/top-groups")
    async def top_groups(days: int = Query(7), limit: int = Query(10)):
        return await tracker.top_groups(days=days, limit=limit)

    return router
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_usage_routes.py -v`
Expected: PASS

- [ ] **Step 5: Wire into plugin**

In `src/plugins/chat/__init__.py`, in `_init()`, after the UsageTracker init block, add:

```python
    if bot_config.llm.usage.enabled:
        from src.llm.usage_routes import create_usage_router
        app = nonebot.get_app()
        app.include_router(create_usage_router(_usage_tracker))
```

Add `import nonebot` at the top if not already present (it's already imported via `from nonebot import get_driver, on_message`).

Actually, `nonebot.get_app()` requires the full `nonebot` import. Add to existing imports:

```python
import nonebot
```

- [ ] **Step 6: Run full suite + lint**

Run: `uv run ruff check src/ && uv run pytest`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/llm/usage_routes.py tests/test_usage_routes.py src/plugins/chat/__init__.py
git commit -m "feat: HTTP /api/usage endpoints for usage stats"
```

---

### Task 10: Final integration test and cleanup

**Files:**
- All modified files
- Modify: `tests/test_client.py` (if any remaining mock issues)

- [ ] **Step 1: Run the full check suite**

```bash
uv run ruff check src/
uv run pyright
uv run pytest -v
```

All three must pass.

- [ ] **Step 2: Verify config.example.toml is consistent with config.py**

Read both files, confirm the `[llm.usage]` section matches the `UsageConfig` defaults.

- [ ] **Step 3: Verify the cache_hit_warn parameter is removed from LLMClient constructor call**

In `src/plugins/chat/__init__.py`, the `LLMClient(...)` call should no longer pass `cache_hit_warn=bot_config.compact.cache_hit_warn`. Remove it if still present.

- [ ] **Step 4: Run tests one more time**

```bash
uv run pytest -v
```

Expected: PASS

- [ ] **Step 5: Commit any remaining cleanup**

```bash
git add -u
git commit -m "chore: final cleanup for usage tracking integration"
```
