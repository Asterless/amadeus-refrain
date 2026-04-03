# Group Compact Cache Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix ~20% prompt cache hit rate in active group chats by removing message eviction and unifying compaction to a single LLM-based trigger.

**Architecture:** Remove `max_timeline_messages` hard truncation that corrupts the Anthropic cache prefix. Replace dual-threshold micro/full compact with a single configurable ratio (70%). Each compact compresses the front 50% of messages via LLM into a summary block that serves as a stable cache anchor.

**Tech Stack:** Python, Pydantic config, pytest

---

### Task 1: Update CompactConfig and GroupConfig

**Files:**
- Modify: `src/config.py:70-82`
- Modify: `src/config.py:44-51`
- Modify: `config.example.toml:82-90`
- Modify: `config.example.toml:108-110`
- Test: `tests/test_config_loader.py`

- [ ] **Step 1: Write failing test for new CompactConfig fields**

In `tests/test_config_loader.py`, add:

```python
def test_compact_config_defaults():
    from src.config import CompactConfig
    c = CompactConfig()
    assert c.ratio == 0.7
    assert c.compress_ratio == 0.5
    assert c.max_failures == 3
    assert c.cache_hit_warn == 90.0


def test_compact_config_rejects_invalid_ratio():
    from src.config import CompactConfig
    import pytest
    with pytest.raises(ValueError):
        CompactConfig(ratio=1.5)
    with pytest.raises(ValueError):
        CompactConfig(compress_ratio=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_loader.py::test_compact_config_defaults -v`
Expected: FAIL — `CompactConfig` still has old fields

- [ ] **Step 3: Update CompactConfig**

In `src/config.py`, replace the `CompactConfig` class (lines 70-82):

```python
class CompactConfig(BaseModel):
    """上下文压缩配置。"""

    ratio: float = 0.7
    compress_ratio: float = 0.5
    max_failures: int = 3
    cache_hit_warn: float = 90.0

    @model_validator(mode="after")
    def _check_ratios(self) -> Self:
        if not (0.0 < self.ratio < 1.0):
            raise ValueError("ratio must be between 0 and 1")
        if not (0.0 < self.compress_ratio < 1.0):
            raise ValueError("compress_ratio must be between 0 and 1")
        return self
```

- [ ] **Step 4: Remove max_timeline_messages from GroupConfig**

In `src/config.py`, change `GroupConfig` (lines 44-51) — remove `max_timeline_messages`:

```python
class GroupConfig(BaseModel):
    """群聊上下文配置。"""

    history_load_count: int = 30
    allowed_groups: list[int] = []
    debounce_seconds: float = 5.0
    batch_size: int = 10
```

- [ ] **Step 5: Update config.example.toml**

In `config.example.toml`, replace the `[compact]` section (lines 82-90):

```toml
[compact]
# compact 触发比例（input_tokens 占 max_context_tokens 的比例）
ratio = 0.7
# 每次压缩前多少比例的消息（0.5 = 前 50%）
compress_ratio = 0.5
# 连续压缩失败上限，超过后跳过
max_failures = 3
# cache_hit 低于此百分比时告警
cache_hit_warn = 90.0
```

Remove `max_timeline_messages` line from `[group]` section (line 110).

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: PASS

- [ ] **Step 7: Run lint and type check**

Run: `uv run ruff check src/config.py && uv run pyright src/config.py`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/config.py config.example.toml tests/test_config_loader.py
git commit -m "refactor: replace dual compact thresholds with single ratio + compress_ratio"
```

---

### Task 2: Remove max_messages from GroupTimeline

**Files:**
- Modify: `src/memory/group_timeline.py:51-59, 62-66, 85-97`
- Modify: `tests/test_group_timeline.py:78-88`
- Modify: `tests/conftest.py:13-14`

- [ ] **Step 1: Write failing test — add() no longer truncates**

In `tests/test_group_timeline.py`, replace `test_max_messages_eviction` (lines 78-88):

```python
def test_add_accumulates_without_limit() -> None:
    """Messages accumulate without hard eviction — compact controls size."""
    tl = GroupTimeline()
    for i in range(500):
        tl.add("g1", role="user", content=f"msg{i}", speaker=f"A({i})")
    msgs = tl.get_messages("g1")
    assert len(msgs) == 500
    assert msgs[0]["content"] == "msg0"
    assert msgs[499]["content"] == "msg499"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_group_timeline.py::test_add_accumulates_without_limit -v`
Expected: FAIL — current code truncates at max_messages

- [ ] **Step 3: Update _GroupState and GroupTimeline**

In `src/memory/group_timeline.py`:

Replace `_GroupState.__init__` (lines 54-59):

```python
class _GroupState:
    __slots__ = ("last_cached_msg_index", "last_input_tokens", "messages", "summary")

    def __init__(self) -> None:
        self.messages: list[TimelineMessage] = []
        self.summary: str = ""
        self.last_input_tokens: int = 0
        self.last_cached_msg_index: int = 0
```

Replace `GroupTimeline.__init__` (lines 62-66):

```python
class GroupTimeline:
    """群聊统一时间线，兼具上下文记录与 compact 能力。"""

    def __init__(self) -> None:
        self._store: dict[str, _GroupState] = {}
```

Update `_get_or_create` (lines 73-79) — remove `_max` reference:

```python
    def _get_or_create(self, group_id: str) -> _GroupState:
        if group_id not in self._store:
            if len(self._store) >= _MAX_GROUPS:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[group_id] = _GroupState()
        return self._store[group_id]
```

Replace `add()` (lines 85-97) — remove truncation:

```python
    def add(
        self,
        group_id: str,
        *,
        role: Literal["user", "assistant"],
        content: Content,
        speaker: str | None = None,
    ) -> None:
        """追加一条消息；由 compact 控制大小，不做硬截断。"""
        state = self._get_or_create(group_id)
        state.messages.append(TimelineMessage(role=role, speaker=speaker, content=content))
```

- [ ] **Step 4: Update conftest.py fixture**

In `tests/conftest.py`, line 14:

```python
@pytest.fixture
def group_timeline() -> GroupTimeline:
    return GroupTimeline()
```

- [ ] **Step 5: Run all GroupTimeline tests**

Run: `uv run pytest tests/test_group_timeline.py -v`
Expected: PASS

- [ ] **Step 6: Lint and type check**

Run: `uv run ruff check src/memory/group_timeline.py && uv run pyright src/memory/group_timeline.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/memory/group_timeline.py tests/test_group_timeline.py tests/conftest.py
git commit -m "refactor: remove max_messages eviction from GroupTimeline"
```

---

### Task 3: Update LLMClient — remove micro compact, use single ratio

**Files:**
- Modify: `src/llm/client.py:253-291` (constructor)
- Modify: `src/llm/client.py:434-450` (chat compact trigger)
- Modify: `src/llm/client.py:598-615` (delete _micro_compact_group)
- Modify: `src/llm/client.py:608-615` (delete _micro_compact_private)
- Modify: `src/llm/client.py:723-740` (_compact_group split calc)
- Modify: `src/llm/client.py:621-640` (_compact split calc)
- Test: `tests/test_client.py`

- [ ] **Step 1: Update test helper _client() to use new params**

In `tests/test_client.py`, replace the `_client` function (lines 52-77) and `timeline` fixture (lines 41-42):

```python
@pytest.fixture
def timeline() -> GroupTimeline:
    return GroupTimeline()


async def _client(
    prompt: PromptBuilder,
    short_term: ShortTermMemory,
    tools: ToolRegistry,
    timeline: GroupTimeline | None = None,
    memo_store: MemoStore | None = None,
    max_compact_failures: int = 3,
) -> AsyncIterator[LLMClient]:
    c = LLMClient(
        base_url="http://fake",
        api_key="sk-fake",
        model="test-model",
        prompt_builder=prompt,
        short_term=short_term,
        tools=tools,
        max_context_tokens=100_000,
        compact_ratio=0.7,
        compress_ratio=0.5,
        max_compact_failures=max_compact_failures,
        group_timeline=timeline,
        memo_store=memo_store,
    )
    try:
        yield c
    finally:
        await c.close()
```

- [ ] **Step 2: Replace micro compact tests with compact-trigger test**

In `tests/test_client.py`, replace the micro compact tests (lines 105-129) with:

```python
# ---------------------------------------------------------------------------
# Compact trigger — single ratio
# ---------------------------------------------------------------------------


async def test_group_compact_triggers_at_ratio(prompt, short_term, tools, timeline, memo_store) -> None:
    """compact_group fires when input_tokens > max_context_tokens * compact_ratio."""
    async for client in _client(prompt, short_term, tools, timeline=timeline, memo_store=memo_store):
        gid = "12345"
        for i in range(8):
            timeline.add(gid, role="user", content=f"msg {i}", speaker=f"user({i})")

        # Simulate previous call reported tokens above threshold (70k > 100k * 0.7)
        timeline.set_input_tokens(gid, 70_001)

        response = json.dumps({"summary": "compressed", "memos": {}, "group_memo": ""})
        mock_compact = {"text": response, "tool_uses": [], "input_tokens": 50}
        mock_chat = {
            "text": "reply", "tool_uses": [], "input_tokens": 5000,
            "output_tokens": 100, "cache_read": 0, "cache_create": 0,
        }

        with patch("src.llm.client._call_api", new_callable=AsyncMock, side_effect=[mock_compact, mock_chat]):
            result = await client.chat(
                session_id="group_12345", user_id="111",
                user_content="hello", identity=_IDENTITY, group_id=gid,
            )

        assert result is not None
        # After compact, old messages should be trimmed
        assert len(timeline.get_messages(gid)) == 4  # 8 * 0.5 = 4 remaining
        assert timeline.get_summary(gid) == "compressed"


async def test_group_no_compact_below_ratio(prompt, short_term, tools, timeline, memo_store) -> None:
    """No compact when tokens below threshold."""
    async for client in _client(prompt, short_term, tools, timeline=timeline, memo_store=memo_store):
        gid = "12345"
        for i in range(8):
            timeline.add(gid, role="user", content=f"msg {i}", speaker=f"user({i})")
        timeline.set_input_tokens(gid, 50_000)  # below 70k threshold

        mock_chat = {
            "text": "reply", "tool_uses": [], "input_tokens": 5000,
            "output_tokens": 100, "cache_read": 0, "cache_create": 0,
        }
        with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_chat):
            await client.chat(
                session_id="group_12345", user_id="111",
                user_content="hello", identity=_IDENTITY, group_id=gid,
            )
        assert len(timeline.get_messages(gid)) == 8  # untouched
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_client.py::test_group_compact_triggers_at_ratio -v`
Expected: FAIL — LLMClient still takes old params

- [ ] **Step 4: Update LLMClient constructor**

In `src/llm/client.py`, replace the `__init__` signature (lines 254-270):

```python
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt_builder: PromptBuilder,
        short_term: ShortTermMemory,
        tools: ToolRegistry,
        max_context_tokens: int = 200_000,
        compact_ratio: float = 0.7,
        compress_ratio: float = 0.5,
        max_compact_failures: int = 3,
        group_timeline: GroupTimeline | None = None,
        memo_store: MemoStore | None = None,
        bot_self_id: str = "",
        on_compact: Callable[[], None] | None = None,
        image_cache: ImageCache | None = None,
    ) -> None:
```

And in the body (lines 286-289), replace:

```python
        self._max_context_tokens = max_context_tokens
        self._compact_ratio = compact_ratio
        self._compress_ratio = compress_ratio
        self._max_compact_failures = max_compact_failures
```

(Delete the old `self._micro_ratio` and `self._full_ratio` lines.)

- [ ] **Step 5: Update chat() compact trigger**

In `src/llm/client.py`, replace the group compact block in `chat()` (lines 438-441):

```python
            if self._timeline.needs_compact(group_id, self._max_context_tokens, self._compact_ratio):
                await self._compact_group(group_id, identity)
```

Delete the `elif` micro compact branch (lines 440-441).

Replace the private chat compact block similarly (lines 446-449):

```python
            if self._short_term.needs_compact(session_id, self._max_context_tokens, self._compact_ratio):
                await self._compact(session_id)
```

Delete the `elif` micro compact branch.

- [ ] **Step 6: Update _compact_group split calculation**

In `src/llm/client.py`, in `_compact_group()` (around line 737), replace:

```python
            split = len(messages) // 2
```

with:

```python
            split = max(2, int(len(messages) * self._compress_ratio))
```

- [ ] **Step 7: Update _compact (private) split calculation**

In `src/llm/client.py`, in `_compact()` (around line 633), replace:

```python
            split = len(history) // 2
```

with:

```python
            split = max(2, int(len(history) * self._compress_ratio))
```

- [ ] **Step 8: Delete _micro_compact_group and _micro_compact_private**

Delete the entire `_micro_compact_group` method (lines 598-606) and `_micro_compact_private` method (lines 608-615).

- [ ] **Step 9: Run all client tests**

Run: `uv run pytest tests/test_client.py -v`
Expected: PASS (micro compact tests were replaced in Step 2)

- [ ] **Step 10: Lint and type check**

Run: `uv run ruff check src/llm/client.py && uv run pyright src/llm/client.py`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add src/llm/client.py tests/test_client.py
git commit -m "refactor: replace micro/full compact with single-ratio compact"
```

---

### Task 4: Update plugin init and config.toml wiring

**Files:**
- Modify: `src/plugins/chat/__init__.py:81, 119-129`
- Modify: `config.example.toml` (already done in Task 1)

- [ ] **Step 1: Update GroupTimeline construction**

In `src/plugins/chat/__init__.py`, line 81, replace:

```python
    _timeline = GroupTimeline(max_messages=bot_config.group.max_timeline_messages)
```

with:

```python
    _timeline = GroupTimeline()
```

- [ ] **Step 2: Update LLMClient construction**

In `src/plugins/chat/__init__.py`, lines 119-129, replace:

```python
    _llm = LLMClient(
        base_url=bot_config.llm.base_url,
        api_key=bot_config.llm.api_key,
        model=bot_config.llm.model,
        prompt_builder=prompt_builder,
        short_term=short_term,
        tools=tools,
        max_context_tokens=bot_config.llm.context.max_context_tokens,
        compact_ratio=bot_config.compact.ratio,
        compress_ratio=bot_config.compact.compress_ratio,
        max_compact_failures=bot_config.compact.max_failures,
        group_timeline=_timeline,
        memo_store=memo_store,
        on_compact=lambda: _dream.notify_compact(),
        image_cache=_image_cache if _vision_enabled else None,
    )
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 4: Lint and type check entire project**

Run: `uv run ruff check src/ && uv run pyright`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/plugins/chat/__init__.py
git commit -m "wire: update plugin init for new compact config"
```

---

### Task 5: Clean up config.toml (gitignored, user reminder)

This task is a manual reminder — `config.toml` is gitignored and must be updated by the user.

- [ ] **Step 1: Print diff for user**

Print the required changes for the user to apply to their `config.toml`:

```
[compact] section:
- Remove: micro_ratio = 0.6
- Remove: full_ratio = 0.8
+ Add: ratio = 0.7
+ Add: compress_ratio = 0.5

[group] section:
- Remove: max_timeline_messages = 200
```

- [ ] **Step 2: Final full test + lint + type check**

Run: `uv run pytest && uv run ruff check src/ && uv run pyright`
Expected: ALL PASS

- [ ] **Step 3: Final commit if any remaining changes**

Verify `git status` is clean or commit any straggling test/doc files.
