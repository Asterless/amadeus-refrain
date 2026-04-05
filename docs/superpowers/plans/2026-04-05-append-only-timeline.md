# Append-Only Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mutable raw message list in `GroupTimeline` with an append-only `_TurnLog` (Anthropic format) + pending buffer + SQLite raw message log, ensuring structural cache prefix stability.

**Architecture:** `_TurnLog` stores finalized Anthropic messages that are never modified after append. A `pending` buffer accumulates raw user messages between bot responses. `add(role="assistant")` flushes pending into a merged user turn + assistant turn. A SQLite `group_messages` table persists every raw message for compact and analysis.

**Tech Stack:** Python 3.12, aiosqlite, pytest

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/memory/group_timeline.py` | **Rewrite** | `_TurnLog`, `_GroupState`, `GroupTimeline` with new turns/pending model |
| `src/memory/message_log.py` | **Create** | `MessageLog` class: SQLite table creation, async `record()`, `query_for_compact()` |
| `src/llm/client.py` | **Modify** | `_build_group_messages` uses `get_turns()`+pending; `_compact_group` queries SQLite |
| `src/plugins/chat/__init__.py` | **Modify** | Initialize `MessageLog`, inject into timeline, close on shutdown |
| `tests/test_group_timeline.py` | **Rewrite** | Tests for new turns/pending/flush lifecycle |
| `tests/test_message_log.py` | **Create** | SQLite record/query tests |
| `tests/test_build_group_messages.py` | **Create** | Cache prefix stability tests |

---

### Task 1: `_TurnLog` Append-Only Container

**Files:**
- Create: `tests/test_group_timeline.py` (replace existing tests)
- Modify: `src/memory/group_timeline.py`

- [ ] **Step 1: Write failing tests for `_TurnLog`**

```python
# tests/test_group_timeline.py
"""GroupTimeline unit tests — turns/pending model."""

import pytest

from src.memory.group_timeline import GroupTimeline, _TurnLog


class TestTurnLog:
    def test_append_and_read(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "hello"})
        assert len(log) == 1
        assert log[0] == {"role": "user", "content": "hello"}

    def test_sequence_protocol(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "a"})
        log.append({"role": "assistant", "content": "b"})
        assert list(log) == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        assert log[0:2] == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    def test_bool_empty(self) -> None:
        log = _TurnLog()
        assert not log
        log.append({"role": "user", "content": "x"})
        assert log

    def test_no_setitem(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "x"})
        with pytest.raises(TypeError):
            log[0] = {"role": "user", "content": "y"}  # type: ignore[index]

    def test_compact_truncate(self) -> None:
        log = _TurnLog()
        for i in range(6):
            log.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
        log.compact_truncate(2)
        assert len(log) == 4
        assert log[0] == {"role": "user", "content": "m2"}

    def test_reset(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "x"})
        log.reset()
        assert len(log) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_group_timeline.py::TestTurnLog -v`
Expected: FAIL — `_TurnLog` not yet defined or import error

- [ ] **Step 3: Implement `_TurnLog`**

Replace the top of `src/memory/group_timeline.py`:

```python
"""群聊统一时间线：append-only turns + pending buffer。"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal, NotRequired, TypedDict, overload

from src.memory.types import Content, ContentBlock, TextBlock

_MAX_GROUPS = 200


class TimelineMessage(TypedDict):
    role: Literal["user", "assistant"]
    speaker: str | None
    content: Content
    message_id: NotRequired[int | None]


class _TurnLog(Sequence[dict[str, Any]]):
    """Append-only store of finalized Anthropic messages.

    Once appended, entries are never modified. Only ``compact_truncate``
    and ``reset`` may remove entries.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: list[dict[str, Any]] = []

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...
    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...
    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def append(self, turn: dict[str, Any]) -> None:
        self._data.append(turn)

    def compact_truncate(self, count: int) -> None:
        """Remove the first *count* entries. Only for compaction."""
        del self._data[:count]

    def reset(self) -> None:
        """Full clear for reconnect reload."""
        self._data.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_group_timeline.py::TestTurnLog -v`
Expected: all 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/group_timeline.py tests/test_group_timeline.py
git commit -m "feat: add _TurnLog append-only container for group timeline"
```

---

### Task 2: `_GroupState` and `GroupTimeline` Core (turns + pending)

**Files:**
- Modify: `tests/test_group_timeline.py`
- Modify: `src/memory/group_timeline.py`

- [ ] **Step 1: Write failing tests for add/flush lifecycle**

Append to `tests/test_group_timeline.py`:

```python
class TestGroupTimelineLifecycle:
    """Tests for the turns + pending flush model."""

    def test_add_user_goes_to_pending(self) -> None:
        tl = GroupTimeline()
        tl.add("g1", role="user", content="hello", speaker="Alice(123)")
        assert len(tl.get_turns("g1")) == 0
        assert len(tl.get_pending("g1")) == 1
        assert tl.get_pending("g1")[0]["content"] == "hello"

    def test_add_assistant_flushes_pending(self) -> None:
        tl = GroupTimeline()
        tl.add("g1", role="user", content="你好", speaker="Alice(123)")
        tl.add("g1", role="user", content="在吗", speaker="Bob(456)")
        tl.add("g1", role="assistant", content="你好！有什么可以帮你？")
        turns = list(tl.get_turns("g1"))
        assert len(turns) == 2
        # First turn: merged user message
        assert turns[0]["role"] == "user"
        assert "Alice(123): 你好" in turns[0]["content"]
        assert "Bob(456): 在吗" in turns[0]["content"]
        # Second turn: assistant
        assert turns[1] == {"role": "assistant", "content": "你好！有什么可以帮你？"}
        # Pending cleared
        assert len(tl.get_pending("g1")) == 0

    def test_add_assistant_without_pending(self) -> None:
        """Assistant with no pending user messages (e.g., history loader)."""
        tl = GroupTimeline()
        tl.add("g1", role="assistant", content="bot message")
        turns = list(tl.get_turns("g1"))
        assert len(turns) == 1
        assert turns[0] == {"role": "assistant", "content": "bot message"}

    def test_multiple_rounds(self) -> None:
        tl = GroupTimeline()
        tl.add("g1", role="user", content="q1", speaker="A(1)")
        tl.add("g1", role="assistant", content="a1")
        tl.add("g1", role="user", content="q2", speaker="B(2)")
        tl.add("g1", role="assistant", content="a2")
        turns = list(tl.get_turns("g1"))
        assert len(turns) == 4
        assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]

    def test_pending_preserved_across_reads(self) -> None:
        """get_pending returns a copy; original is not affected."""
        tl = GroupTimeline()
        tl.add("g1", role="user", content="msg", speaker="A(1)")
        pending = tl.get_pending("g1")
        pending.clear()  # mutating the copy
        assert len(tl.get_pending("g1")) == 1  # original unchanged

    def test_group_isolation(self) -> None:
        tl = GroupTimeline()
        tl.add("g1", role="user", content="群1", speaker="A(1)")
        tl.add("g2", role="user", content="群2", speaker="B(2)")
        assert len(tl.get_pending("g1")) == 1
        assert len(tl.get_pending("g2")) == 1
        assert tl.get_pending("g1")[0]["content"] == "群1"

    def test_turn_times_recorded(self) -> None:
        tl = GroupTimeline()
        tl.add("g1", role="user", content="q", speaker="A(1)")
        tl.add("g1", role="assistant", content="a")
        assert tl.get_turn_time("g1", 0) > 0
        assert tl.get_turn_time("g1", 1) > 0
        assert tl.get_turn_time("g1", 1) >= tl.get_turn_time("g1", 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_group_timeline.py::TestGroupTimelineLifecycle -v`
Expected: FAIL — `get_turns`, `get_pending` not defined

- [ ] **Step 3: Implement `_GroupState` and rewrite `GroupTimeline`**

Rewrite the rest of `src/memory/group_timeline.py` below `_TurnLog`:

```python
def _merge_user_contents(batch: list[TimelineMessage]) -> Content:
    """Merge consecutive user messages into a single content value."""
    # ... keep existing implementation unchanged ...


def _merge_assistant_contents(parts: list[Content]) -> Content:
    """Merge consecutive assistant message contents into one."""
    # ... keep existing implementation unchanged ...


class _GroupState:
    __slots__ = (
        "last_cached_msg_index",
        "last_input_tokens",
        "pending",
        "summary",
        "turn_times",
        "turns",
    )

    def __init__(self) -> None:
        self.turns = _TurnLog()
        self.turn_times: list[float] = []
        self.pending: list[TimelineMessage] = []
        self.summary: str = ""
        self.last_input_tokens: int = 0
        self.last_cached_msg_index: int = 0


class GroupTimeline:
    """群聊统一时间线：append-only turns + pending buffer。"""

    def __init__(self) -> None:
        self._store: dict[str, _GroupState] = {}

    def _get_or_create(self, group_id: str) -> _GroupState:
        if group_id not in self._store:
            if len(self._store) >= _MAX_GROUPS:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[group_id] = _GroupState()
        return self._store[group_id]

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add(
        self,
        group_id: str,
        *,
        role: Literal["user", "assistant"],
        content: Content,
        speaker: str | None = None,
        message_id: int | None = None,
    ) -> None:
        """Add a message. user→pending; assistant→flush pending+append turns."""
        state = self._get_or_create(group_id)

        if role == "user":
            msg = TimelineMessage(role=role, speaker=speaker, content=content)
            if message_id is not None:
                msg["message_id"] = message_id
            state.pending.append(msg)
        else:
            # Flush pending user messages into a merged user turn
            now = time.time()
            if state.pending:
                merged = _merge_user_contents(state.pending)
                state.turns.append({"role": "user", "content": merged})
                state.turn_times.append(now)
                state.pending.clear()
            # Append assistant turn
            state.turns.append({"role": "assistant", "content": content})
            state.turn_times.append(now)

    def get_turns(self, group_id: str) -> Sequence[dict[str, Any]]:
        """Read-only view of finalized Anthropic messages."""
        if group_id not in self._store:
            return ()
        return self._store[group_id].turns

    def get_pending(self, group_id: str) -> list[TimelineMessage]:
        """Copy of pending user message buffer."""
        if group_id not in self._store:
            return []
        return list(self._store[group_id].pending)

    def clear(self, group_id: str) -> None:
        """Clear all turns and pending for a group, preserving summary."""
        if group_id in self._store:
            state = self._store[group_id]
            state.turns.reset()
            state.turn_times.clear()
            state.pending.clear()
            state.last_input_tokens = 0
            state.last_cached_msg_index = 0

    # ------------------------------------------------------------------
    # Summary & token management
    # ------------------------------------------------------------------

    def get_summary(self, group_id: str) -> str:
        if group_id not in self._store:
            return ""
        return self._store[group_id].summary

    def set_input_tokens(self, group_id: str, tokens: int) -> None:
        state = self._get_or_create(group_id)
        state.last_input_tokens = tokens

    def get_input_tokens(self, group_id: str) -> int:
        if group_id not in self._store:
            return 0
        return self._store[group_id].last_input_tokens

    def get_cached_msg_index(self, group_id: str) -> int:
        if group_id not in self._store:
            return 0
        return self._store[group_id].last_cached_msg_index

    def set_cached_msg_index(self, group_id: str, index: int) -> None:
        state = self._get_or_create(group_id)
        state.last_cached_msg_index = index

    def needs_compact(self, group_id: str, max_tokens: int, ratio: float) -> bool:
        return self.get_input_tokens(group_id) > max_tokens * ratio

    def compact(self, group_id: str, split: int, new_summary: str) -> None:
        """Truncate first *split* turns, update summary, reset token state."""
        if group_id not in self._store:
            return
        state = self._store[group_id]
        state.turns.compact_truncate(split)
        state.turn_times = state.turn_times[split:]
        state.summary = new_summary
        state.last_input_tokens = 0
        state.last_cached_msg_index = 0

    def get_turn_time(self, group_id: str, index: int) -> float:
        """Return created_at timestamp for a specific turn index."""
        return self._store[group_id].turn_times[index]

    def drop_oldest(self, group_id: str, count: int) -> None:
        """Drop the oldest *count* turns. For circuit breaker."""
        state = self._store.get(group_id)
        if state is None:
            return
        state.turns.compact_truncate(count)
        state.turn_times = state.turn_times[count:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_group_timeline.py -v`
Expected: all tests PASS (both `TestTurnLog` and `TestGroupTimelineLifecycle`)

- [ ] **Step 5: Run lint and type check**

Run: `uv run ruff check src/memory/group_timeline.py && uv run pyright src/memory/group_timeline.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/memory/group_timeline.py tests/test_group_timeline.py
git commit -m "feat: rewrite GroupTimeline with turns/pending flush model"
```

---

### Task 3: `MessageLog` — SQLite Raw Message Persistence

**Files:**
- Create: `src/memory/message_log.py`
- Create: `tests/test_message_log.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_message_log.py
"""MessageLog SQLite persistence tests."""

import pytest

from src.memory.message_log import MessageLog


@pytest.fixture
async def msg_log(tmp_path) -> MessageLog:
    log = MessageLog(db_path=str(tmp_path / "test_messages.db"))
    await log.init()
    yield log
    await log.close()


@pytest.mark.asyncio
async def test_record_and_query(msg_log: MessageLog) -> None:
    await msg_log.record(
        group_id="g1",
        role="user",
        speaker="Alice(123)",
        content_text="你好",
        content_json='"你好"',
        message_id=100,
    )
    await msg_log.record(
        group_id="g1",
        role="assistant",
        speaker=None,
        content_text="你好！",
        content_json='"你好！"',
        message_id=None,
    )
    rows = await msg_log.query_for_compact("g1", before=9999999999.0)
    assert len(rows) == 2
    assert rows[0]["role"] == "user"
    assert rows[0]["speaker"] == "Alice(123)"
    assert rows[0]["content_text"] == "你好"
    assert rows[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_query_respects_time_bound(msg_log: MessageLog) -> None:
    import time

    t_before = time.time()
    await msg_log.record(
        group_id="g1", role="user", speaker="A(1)",
        content_text="old", content_json='"old"',
    )
    t_mid = time.time()
    await msg_log.record(
        group_id="g1", role="user", speaker="B(2)",
        content_text="new", content_json='"new"',
    )
    rows = await msg_log.query_for_compact("g1", before=t_mid)
    assert len(rows) == 1
    assert rows[0]["content_text"] == "old"


@pytest.mark.asyncio
async def test_group_isolation(msg_log: MessageLog) -> None:
    await msg_log.record(
        group_id="g1", role="user", speaker="A(1)",
        content_text="群1", content_json='"群1"',
    )
    await msg_log.record(
        group_id="g2", role="user", speaker="B(2)",
        content_text="群2", content_json='"群2"',
    )
    rows = await msg_log.query_for_compact("g1", before=9999999999.0)
    assert len(rows) == 1
    assert rows[0]["content_text"] == "群1"


@pytest.mark.asyncio
async def test_content_json_with_blocks(msg_log: MessageLog) -> None:
    import json

    blocks = [
        {"type": "text", "text": "看这个"},
        {"type": "image_ref", "path": "storage/image_cache/ab/abc.jpg", "media_type": "image/jpeg"},
    ]
    await msg_log.record(
        group_id="g1", role="user", speaker="A(1)",
        content_text="看这个",
        content_json=json.dumps(blocks, ensure_ascii=False),
    )
    rows = await msg_log.query_for_compact("g1", before=9999999999.0)
    assert rows[0]["content_text"] == "看这个"
    parsed = json.loads(rows[0]["content_json"])
    assert parsed[1]["type"] == "image_ref"
    assert parsed[1]["path"] == "storage/image_cache/ab/abc.jpg"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_message_log.py -v`
Expected: FAIL — `src.memory.message_log` not found

- [ ] **Step 3: Implement `MessageLog`**

```python
# src/memory/message_log.py
"""SQLite persistence for raw group chat messages."""

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

_CREATE_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_gm_group_time "
    "ON group_messages(group_id, created_at)"
)

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
    """Async SQLite store for raw group chat messages."""

    def __init__(self, db_path: str = "storage/messages.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_INDEX)
        await self._db.commit()

    async def close(self) -> None:
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
        if not self._db:
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
        self, group_id: str, *, before: float,
    ) -> list[dict[str, Any]]:
        """Return raw messages for compact: ordered by time, up to cutoff."""
        if not self._db:
            return []
        cursor = await self._db.execute(_QUERY_FOR_COMPACT, (group_id, before))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_message_log.py -v`
Expected: all 4 PASS

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check src/memory/message_log.py && uv run pyright src/memory/message_log.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/memory/message_log.py tests/test_message_log.py
git commit -m "feat: add MessageLog SQLite store for raw group messages"
```

---

### Task 4: Wire `MessageLog` into `GroupTimeline.add()`

**Files:**
- Modify: `src/memory/group_timeline.py`
- Modify: `tests/test_group_timeline.py`

- [ ] **Step 1: Write failing test for SQLite integration**

Append to `tests/test_group_timeline.py`:

```python
@pytest.mark.asyncio
async def test_add_writes_to_message_log(tmp_path) -> None:
    """add() fires SQLite writes when message_log is attached."""
    from src.memory.message_log import MessageLog

    log = MessageLog(db_path=str(tmp_path / "test.db"))
    await log.init()
    tl = GroupTimeline(message_log=log)
    tl.add("g1", role="user", content="hello", speaker="Alice(123)", message_id=42)
    tl.add("g1", role="assistant", content="hi there")

    # Give fire-and-forget tasks a moment to complete
    import asyncio
    await asyncio.sleep(0.1)

    rows = await log.query_for_compact("g1", before=9999999999.0)
    assert len(rows) == 2
    assert rows[0]["speaker"] == "Alice(123)"
    assert rows[0]["content_text"] == "hello"
    assert rows[0]["message_id"] == 42
    assert rows[1]["role"] == "assistant"
    assert rows[1]["content_text"] == "hi there"
    await log.close()


@pytest.mark.asyncio
async def test_add_multimodal_writes_both_columns(tmp_path) -> None:
    """Image content writes content_text (text only) and content_json (full)."""
    import asyncio
    import json

    from src.memory.message_log import MessageLog
    from src.memory.types import ImageRefBlock, TextBlock

    log = MessageLog(db_path=str(tmp_path / "test.db"))
    await log.init()
    tl = GroupTimeline(message_log=log)
    blocks = [
        TextBlock(type="text", text="看这个"),
        ImageRefBlock(type="image_ref", path="storage/img.jpg", media_type="image/jpeg"),
    ]
    tl.add("g1", role="user", content=blocks, speaker="A(1)")
    await asyncio.sleep(0.1)

    rows = await log.query_for_compact("g1", before=9999999999.0)
    assert rows[0]["content_text"] == "看这个"
    parsed = json.loads(rows[0]["content_json"])
    assert parsed[1]["type"] == "image_ref"
    assert parsed[1]["path"] == "storage/img.jpg"
    await log.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_group_timeline.py::test_add_writes_to_message_log -v`
Expected: FAIL — `GroupTimeline.__init__` doesn't accept `message_log`

- [ ] **Step 3: Add `message_log` integration to `GroupTimeline`**

In `src/memory/group_timeline.py`, add imports and modify `GroupTimeline.__init__` and `add()`:

```python
# Add imports at top
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, overload

from src.memory.types import Content, ContentBlock, TextBlock

if TYPE_CHECKING:
    from src.memory.message_log import MessageLog
```

Modify `GroupTimeline.__init__`:

```python
class GroupTimeline:
    """群聊统一时间线：append-only turns + pending buffer。"""

    def __init__(self, message_log: MessageLog | None = None) -> None:
        self._store: dict[str, _GroupState] = {}
        self._message_log = message_log
```

Add a helper to extract text from `Content`:

```python
    @staticmethod
    def _content_to_text(content: Content) -> str:
        """Extract plain text from Content for SQLite content_text column."""
        if isinstance(content, str):
            return content
        return " ".join(b["text"] for b in content if b.get("type") == "text")

    @staticmethod
    def _content_to_json(content: Content) -> str:
        """Serialize Content to JSON for SQLite content_json column."""
        if isinstance(content, str):
            return json.dumps(content, ensure_ascii=False)
        return json.dumps(content, ensure_ascii=False)
```

In `add()`, after appending to pending or turns, fire SQLite write:

```python
    def add(self, group_id, *, role, content, speaker=None, message_id=None):
        state = self._get_or_create(group_id)

        if role == "user":
            msg = TimelineMessage(role=role, speaker=speaker, content=content)
            if message_id is not None:
                msg["message_id"] = message_id
            state.pending.append(msg)
        else:
            now = time.time()
            if state.pending:
                merged = _merge_user_contents(state.pending)
                state.turns.append({"role": "user", "content": merged})
                state.turn_times.append(now)
                state.pending.clear()
            state.turns.append({"role": "assistant", "content": content})
            state.turn_times.append(now)

        # Fire-and-forget SQLite write
        if self._message_log:
            asyncio.create_task(self._message_log.record(  # noqa: RUF006
                group_id=group_id,
                role=role,
                speaker=speaker,
                content_text=self._content_to_text(content),
                content_json=self._content_to_json(content),
                message_id=message_id,
            ))
```

- [ ] **Step 4: Run all timeline tests**

Run: `uv run pytest tests/test_group_timeline.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check src/memory/group_timeline.py && uv run pyright src/memory/group_timeline.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/memory/group_timeline.py tests/test_group_timeline.py
git commit -m "feat: wire MessageLog SQLite writes into GroupTimeline.add()"
```

---

### Task 5: Update `_build_group_messages` in `client.py`

**Files:**
- Create: `tests/test_build_group_messages.py`
- Modify: `src/llm/client.py`

- [ ] **Step 1: Write failing test for new build logic**

```python
# tests/test_build_group_messages.py
"""Test _build_group_messages uses turns + pending correctly."""

from src.memory.group_timeline import GroupTimeline, _merge_user_contents


class TestBuildGroupMessages:
    """Verify the Anthropic message list structure from turns + pending."""

    def test_turns_are_included_directly(self) -> None:
        tl = GroupTimeline()
        tl.add("g1", role="user", content="q1", speaker="A(1)")
        tl.add("g1", role="assistant", content="a1")
        turns = list(tl.get_turns("g1"))
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

    def test_pending_merged_as_tail(self) -> None:
        tl = GroupTimeline()
        tl.add("g1", role="user", content="q1", speaker="A(1)")
        tl.add("g1", role="assistant", content="a1")
        tl.add("g1", role="user", content="q2", speaker="B(2)")
        tl.add("g1", role="user", content="q3", speaker="C(3)")

        turns = list(tl.get_turns("g1"))
        pending = tl.get_pending("g1")
        assert len(turns) == 2
        assert len(pending) == 2

        # Simulate what _build_group_messages does
        messages = list(turns)
        if pending:
            messages.append({"role": "user", "content": _merge_user_contents(pending)})
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert "B(2): q2" in messages[2]["content"]
        assert "C(3): q3" in messages[2]["content"]

    def test_prefix_stability_across_appends(self) -> None:
        """Core cache invariant: turns prefix doesn't change when pending grows."""
        tl = GroupTimeline()
        tl.add("g1", role="user", content="q1", speaker="A(1)")
        tl.add("g1", role="assistant", content="a1")

        snapshot_before = [dict(t) for t in tl.get_turns("g1")]

        # More user messages arrive
        tl.add("g1", role="user", content="q2", speaker="B(2)")
        tl.add("g1", role="user", content="q3", speaker="C(3)")

        snapshot_after = [dict(t) for t in tl.get_turns("g1")]
        assert snapshot_before == snapshot_after  # Turns unchanged

    def test_prefix_stability_after_flush(self) -> None:
        """After flush, old turns are still identical objects."""
        tl = GroupTimeline()
        tl.add("g1", role="user", content="q1", speaker="A(1)")
        tl.add("g1", role="assistant", content="a1")

        old_turns = list(tl.get_turns("g1"))

        tl.add("g1", role="user", content="q2", speaker="B(2)")
        tl.add("g1", role="assistant", content="a2")

        new_turns = list(tl.get_turns("g1"))
        # First two turns are the exact same objects (not just equal)
        assert new_turns[0] is old_turns[0]
        assert new_turns[1] is old_turns[1]
```

- [ ] **Step 2: Run tests to verify they pass** (these test the timeline, not client.py yet)

Run: `uv run pytest tests/test_build_group_messages.py -v`
Expected: all PASS (tests exercise GroupTimeline directly)

- [ ] **Step 3: Update `_build_group_messages` in `client.py`**

Replace `_build_group_messages` (around line 424-458) with:

```python
    def _build_group_messages(self, group_id: str) -> list[dict[str, Any]]:
        """Build message list for group chat: optional summary + turns + pending + cache breakpoint."""
        assert self._timeline is not None
        messages: list[dict[str, Any]] = []

        # Summary as stable prefix for cache hits
        summary = self._timeline.get_summary(group_id)
        if summary:
            messages.append({
                "role": "user",
                "content": [_cached_text(f"«对话摘要»\n{summary}")],
            })
            messages.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

        # Turns — finalized, byte-identical to previous API calls
        messages.extend(self._timeline.get_turns(group_id))

        # Pending — temporary merge as tail user message
        pending = self._timeline.get_pending(group_id)
        if pending:
            from src.memory.group_timeline import _merge_user_contents
            messages.append({"role": "user", "content": _merge_user_contents(pending)})

        # Place cache breakpoint at the position recorded by the previous API call
        cached_idx = self._timeline.get_cached_msg_index(group_id)
        if 0 < cached_idx < len(messages):
            target = messages[cached_idx]
            content = target.get("content")
            if isinstance(content, str):
                messages[cached_idx] = {"role": target["role"], "content": [_cached_text(content)]}
            elif isinstance(content, list):
                content = [*content]
                content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
                messages[cached_idx] = {"role": target["role"], "content": content}

        # Store second-to-last for next call (last may grow with new pending)
        if len(messages) >= 2:
            self._timeline.set_cached_msg_index(group_id, len(messages) - 2)

        return messages
```

- [ ] **Step 4: Remove old `to_anthropic_messages` and `get_messages` from `GroupTimeline`**

In `src/memory/group_timeline.py`, delete the `to_anthropic_messages` method and the `get_messages` method. They are no longer used.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS. If old tests reference `get_messages` or `to_anthropic_messages`, they were already replaced in Task 1/2.

- [ ] **Step 6: Lint and type check**

Run: `uv run ruff check src/ && uv run pyright src/llm/client.py`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add src/llm/client.py src/memory/group_timeline.py tests/
git commit -m "feat: _build_group_messages uses turns + pending, remove to_anthropic_messages"
```

---

### Task 6: Update `_compact_group` to Use SQLite

**Files:**
- Modify: `src/llm/client.py`

- [ ] **Step 1: Update `LLMClient` to accept `MessageLog`**

In `client.py`, add to imports:

```python
from src.memory.message_log import MessageLog
```

In `LLMClient.__init__` (around line 340), add parameter:

```python
        message_log: MessageLog | None = None,
```

And store it:

```python
        self._message_log = message_log
```

- [ ] **Step 2: Rewrite `_compact_group` to query SQLite**

Replace the `_compact_group` method (around line 867). Key changes:
- Use `self._message_log.query_for_compact()` instead of `self._timeline.get_messages()`
- Build conversation text from SQLite rows' `content_text` and `speaker` columns
- Extract QQ IDs from `speaker` column
- `split` operates on turns (not raw messages)

```python
    async def _compact_group(self, group_id: str, identity: Identity) -> None:
        """Compress first half of group turns into summary and extract memos."""
        if self._group_compact_failures >= self._max_compact_failures:
            assert self._timeline is not None
            turns = self._timeline.get_turns(group_id)
            drop = max(2, int(len(turns) * self._compress_ratio))
            self._timeline.drop_oldest(group_id, drop)
            logger.warning("compact circuit breaker active, dropped {} turns | group={}", drop, group_id)
            return

        try:
            assert self._timeline is not None
            turns = self._timeline.get_turns(group_id)
            if len(turns) < 4:
                return

            old_summary = self._timeline.get_summary(group_id)
            split = max(2, int(len(turns) * self._compress_ratio))

            # Build conversation text from SQLite raw messages
            lines: list[str] = []
            seen_user_ids: list[str] = []
            seen: set[str] = set()

            if old_summary:
                lines.append(f"«之前的对话摘要»\n{old_summary}\n")

            if self._message_log:
                # Query raw messages up to the time of the last turn being compacted
                cutoff = self._timeline.get_turn_time(group_id, split - 1)
                rows = await self._message_log.query_for_compact(group_id, before=cutoff)
                for row in rows:
                    speaker = row.get("speaker") or ""
                    text = row.get("content_text") or ""
                    if row["role"] == "assistant":
                        lines.append(f"{identity.name}: {text}")
                    elif speaker:
                        lines.append(f"{speaker}: {text}")
                    else:
                        lines.append(f"用户: {text}")
                    # Extract QQ IDs
                    if speaker and self._memo_store:
                        m = re.search(r"\((\d+)\)$", speaker)
                        if m and m.group(1) not in seen:
                            seen.add(m.group(1))
                            seen_user_ids.append(m.group(1))
            else:
                # Fallback: reconstruct from turns content (no speaker info)
                for turn in turns[:split]:
                    text = _content_text(turn.get("content", ""))
                    if turn["role"] == "assistant":
                        lines.append(f"{identity.name}: {text}")
                    else:
                        lines.append(f"用户: {text}")

            conversation_text = "\n".join(lines)

            system = [{"type": "text", "text": (
                "你是一个对话分析助手。请完成两个任务：\n"
                "1. 将以下群聊记录压缩成简洁的中文摘要。保留关键信息。\n"
                "2. 如果对话中出现了关于用户或群组的新信息，"
                "用 append_memo 工具追加新观察。每条 note 写一句话结论，"
                "系统会自动放入「待整理」区域。没有新信息则不需要调用。\n\n"
                "**关键规则——个人情报与群情报分离：**\n"
                "- 关于某个人的信息（性格、偏好、背景、身份、观点、与他人的关系）"
                "→ 写入该用户的备忘录（user_QQ号）\n"
                "- 只有群级别信息（群氛围、群事件、群规矩、成员变动）"
                "→ 写入群备忘录（group_群号）\n"
                "- 判断标准：如果信息跟着这个人走（换个群也成立），写 user_；"
                "如果只在本群语境下有意义，写 group_\n\n"
                f"本群 ID: group_{group_id}\n"
                f"出现的用户 ID: {', '.join(f'user_{uid}' for uid in seen_user_ids)}\n\n"
                "备忘录规则：用 @QQ号 标注人物，#群号 标注群。QQ号是唯一身份标识，昵称不可信。\n"
                "只记新的印象和结论，不记流水账。\n"
                "最终请输出纯摘要文本（不要加标题或格式）。"
            )}]
            compress_messages: list[dict[str, Any]] = [{"role": "user", "content": conversation_text}]

            logger.info("compact_group | group={} split={}/{}", group_id, split, len(turns))
            source = f"compact:group:{group_id}"
            t_compact = time.monotonic()
            new_summary, memo_writes = await self._compact_with_tools(
                system, compress_messages, source, group_id=group_id,
            )
            compact_elapsed = time.monotonic() - t_compact

            if new_summary:
                self._timeline.compact(group_id, split, new_summary)
                logger.info(
                    "compact_group done | group={} turns={}->{} summary_len={} memo_writes={} elapsed={:.1f}s",
                    group_id, len(turns), len(turns) - split,
                    len(new_summary), memo_writes, compact_elapsed,
                )
            else:
                logger.warning("compact_group produced empty summary | group={}", group_id)

        except RateLimitError:
            logger.warning("compact_group rate limited | group={}", group_id)
        except Exception:
            self._group_compact_failures += 1
            logger.exception("compact_group failed ({}/{})", self._group_compact_failures, self._max_compact_failures)
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 4: Lint and type check**

Run: `uv run ruff check src/llm/client.py && uv run pyright src/llm/client.py`
Expected: clean

- [ ] **Step 5: Commit**

```bash
git add src/llm/client.py
git commit -m "feat: _compact_group queries SQLite for raw messages with speaker info"
```

---

### Task 7: Wire `MessageLog` in Plugin Startup/Shutdown

**Files:**
- Modify: `src/plugins/chat/__init__.py`

- [ ] **Step 1: Initialize `MessageLog` in `_startup`**

After the `_usage_tracker` init block (around line 152), add:

```python
    from src.memory.message_log import MessageLog

    _message_log = MessageLog(db_path="storage/messages.db")
    await _message_log.init()
```

Pass it to `GroupTimeline` (modify the existing `_timeline = GroupTimeline()` line, around line 120):

```python
    _timeline = GroupTimeline(message_log=_message_log)
```

Pass it to `LLMClient` constructor (add to the `LLMClient(...)` call around line 156):

```python
        message_log=_message_log,
```

- [ ] **Step 2: Add module-level declaration**

Near the top of the module with other globals:

```python
_message_log: MessageLog
```

(Add the import with other TYPE_CHECKING imports or directly.)

- [ ] **Step 3: Close on shutdown**

In `_shutdown()` (around line 192), add before `await _usage_tracker.close()`:

```python
    await _message_log.close()
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check src/plugins/chat/__init__.py && uv run pyright src/plugins/chat/__init__.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/plugins/chat/__init__.py
git commit -m "feat: wire MessageLog into plugin startup/shutdown lifecycle"
```

---

### Task 8: Clean Up Old Tests and Final Verification

**Files:**
- Modify: `tests/conftest.py`
- Verify: all test files

- [ ] **Step 1: Update conftest fixture**

The `group_timeline` fixture in `tests/conftest.py` should still work (no `message_log` arg means `None`). Verify it:

```python
@pytest.fixture
def group_timeline() -> GroupTimeline:
    return GroupTimeline()
```

No change needed — `GroupTimeline(message_log=None)` is the default.

- [ ] **Step 2: Check for any remaining references to removed methods**

Search for `get_messages`, `to_anthropic_messages` in test files and source:

Run: `uv run ruff check src/ && grep -r "get_messages\|to_anthropic_messages" src/ tests/ --include="*.py"`

Fix any remaining references. `get_messages` should be replaced with `get_turns` or `get_pending` as appropriate.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: all PASS

- [ ] **Step 4: Run full lint + type check**

Run: `uv run ruff check src/ && uv run pyright`
Expected: clean

- [ ] **Step 5: Commit any cleanup**

```bash
git add -u
git commit -m "chore: clean up references to removed timeline methods"
```

- [ ] **Step 6: Final integration check**

Run: `uv run ruff check src/ && uv run pyright && uv run pytest -v`
Expected: all clean, all pass
