# Unified Chat Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge ProactiveEvaluator + warm_cache + chat() into a single flow where the main model decides to reply (text) or skip (`pass_turn` tool).

**Architecture:** New `GroupChatScheduler` handles debounce/batch/interrupt. `LLMClient.chat()` gains `allow_skip` param and `pass_turn` built-in tool. ProactiveEvaluator and warm_cache are deleted.

**Tech Stack:** Python 3.12, asyncio, aiohttp, NoneBot2, Anthropic API

**Spec:** `docs/superpowers/specs/2026-04-02-unified-chat-scheduler-design.md`

---

## File Map

| Operation | File | Responsibility |
|-----------|------|---------------|
| Create | `src/llm/scheduler.py` | GroupChatScheduler: debounce, batch, interrupt, _do_chat |
| Create | `tests/test_scheduler.py` | Scheduler unit tests |
| Modify | `src/llm/client.py` | Add pass_turn + allow_skip, delete warm_cache/proactive_hint |
| Modify | `src/llm/prompt.py` | Inject identity.proactive into system blocks for group chat |
| Modify | `src/config.py` | Delete CacheConfig/ProactiveConfig, add debounce fields to GroupConfig |
| Modify | `src/config_loader.py` | Delete PROACTIVE_MODEL env mapping |
| Modify | `src/memory/group_timeline.py` | Delete warm-related fields/methods |
| Modify | `src/plugins/chat/__init__.py` | Replace ProactiveEvaluator+warm with GroupChatScheduler |
| Modify | `config.toml` | Delete [llm.cache] and [proactive], add debounce fields to [group] |
| Delete | `src/llm/proactive.py` | Entire file |
| Delete | `tests/test_proactive.py` | Entire file |
| Modify | `tests/test_group_timeline.py` | Delete 4 warm-related tests |
| Modify | `tests/test_config_loader.py` | Delete proactive/cache config assertions |
| Modify | `tests/test_prompt.py` | Add test for proactive injection |

---

### Task 1: Config cleanup — delete CacheConfig/ProactiveConfig, add debounce fields

**Files:**
- Modify: `src/config.py`
- Modify: `src/config_loader.py`
- Modify: `config.toml`
- Modify: `tests/test_config_loader.py`

- [ ] **Step 1: Update `src/config.py`**

Delete `CacheConfig` class (lines 6-11), delete `ProactiveConfig` class (lines 50-59), remove `cache` field from `LLMConfig`, remove `proactive` field from `BotConfig`, add `debounce_seconds` and `batch_size` to `GroupConfig`:

```python
# src/config.py — full new content:

"""Bot 配置：嵌套 Pydantic 模型，支持 TOML / 环境变量 / CLI 覆盖。"""

from pydantic import BaseModel


class ContextConfig(BaseModel):
    """上下文窗口与压缩配置。"""

    max_context_tokens: int = 1_000_000
    compact_ratio: float = 0.7


class LLMConfig(BaseModel):
    """LLM 接入配置。"""

    base_url: str = "http://127.0.0.1:34567/v1"
    api_key: str = "sk-placeholder"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    context: ContextConfig = ContextConfig()


class LogConfig(BaseModel):
    """日志配置。"""

    dir: str = "storage/logs"


class MemoryConfig(BaseModel):
    """长期记忆存储配置。"""

    dir: str = "storage/memories"


class SoulConfig(BaseModel):
    """人设与指令配置目录。"""

    dir: str = "soul"


class GroupConfig(BaseModel):
    """群聊上下文配置。"""

    max_timeline_messages: int = 200
    history_load_count: int = 30
    allowed_groups: list[int] = []
    debounce_seconds: float = 5.0
    batch_size: int = 10


class NapcatConfig(BaseModel):
    """NapCat HTTP API 配置。"""

    api_url: str = "http://localhost:29300"


class BotConfig(BaseModel):
    """全局 Bot 配置。"""

    llm: LLMConfig = LLMConfig()
    log: LogConfig = LogConfig()
    memory: MemoryConfig = MemoryConfig()
    soul: SoulConfig = SoulConfig()
    group: GroupConfig = GroupConfig()
    napcat: NapcatConfig = NapcatConfig()
    superusers: set[str] = set()
    allowed_private_users: list[int] = []
```

- [ ] **Step 2: Update `src/config_loader.py`**

Delete `"PROACTIVE_MODEL": "proactive.model"` from `_ENV_MAP`:

```python
_ENV_MAP: dict[str, str] = {
    "LLM_BASE_URL": "llm.base_url",
    "LLM_API_KEY": "llm.api_key",
    "LLM_MODEL": "llm.model",
    "NAPCAT_API_URL": "napcat.api_url",
}
```

- [ ] **Step 3: Update `config.toml`**

Delete the `[llm.cache]` section (lines 43-54) and the `[proactive]` section (lines 97-120). Add debounce fields to `[group]`:

```toml
[group]
# 内存中保留的最大群消息条数
max_timeline_messages = 200

# 启动时从 NapCat 拉取的历史消息条数
history_load_count = 30

# 群聊白名单，只处理这些群的消息。空数组 = 不限制
allowed_groups = [100001, 100002]

# 非@消息的 debounce 等待时间（秒），等消息间隙后触发模型调用
debounce_seconds = 5.0

# 最多攒多少条消息强制触发模型调用
batch_size = 10
```

- [ ] **Step 4: Update `tests/test_config_loader.py`**

In `test_load_defaults_without_file`: delete the 3 lines asserting `cfg.llm.cache.*`. Add assertions for new defaults:

```python
    assert cfg.group.debounce_seconds == 5.0
    assert cfg.group.batch_size == 10
```

In `test_load_from_toml`: delete the 3 lines asserting `cfg.llm.cache.*`, delete the `[llm.cache]` section from the TOML string. Add to the TOML string inside `[group]`:

```toml
debounce_seconds = 3.0
batch_size = 5
```

And add assertions:

```python
    assert cfg.group.debounce_seconds == 3.0
    assert cfg.group.batch_size == 5
```

Delete `test_proactive_config_defaults` and `test_proactive_config_from_toml` entirely.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: all pass

- [ ] **Step 6: Run lint and type check**

Run: `uv run ruff check src/config.py src/config_loader.py && uv run pyright src/config.py src/config_loader.py`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add src/config.py src/config_loader.py config.toml tests/test_config_loader.py
git commit -m "refactor: delete CacheConfig/ProactiveConfig, add debounce fields to GroupConfig"
```

---

### Task 2: GroupTimeline — delete warm-related fields and methods

**Files:**
- Modify: `src/memory/group_timeline.py`
- Modify: `tests/test_group_timeline.py`

- [ ] **Step 1: Update `src/memory/group_timeline.py`**

In `_GroupState.__slots__`, remove `"last_api_call_time"` and `"new_msg_count"`. In `__init__`, remove the corresponding assignments.

In `_GroupState.__init__`, the result should be:

```python
    def __init__(self, max_messages: int) -> None:
        self._max = max_messages
        self.messages: list[TimelineMessage] = []
        self.summary: str = ""
        self.last_input_tokens: int = 0
```

And `__slots__`:

```python
    __slots__ = ("_max", "last_input_tokens", "messages", "summary")
```

In `add()`, remove the line `if role == "user": state.new_msg_count += 1`.

Simplify `set_input_tokens` — remove `last_api_call_time` and `new_msg_count` writes:

```python
    def set_input_tokens(self, group_id: str, tokens: int) -> None:
        """Record input token count from the latest API call."""
        state = self._get_or_create(group_id)
        state.last_input_tokens = tokens
```

Delete `should_warm()` method entirely (lines 145-161).

Delete `reset_warm_counter()` method entirely (lines 163-165).

- [ ] **Step 2: Update `tests/test_group_timeline.py`**

Delete these 4 test functions entirely:
- `test_should_warm_basic`
- `test_should_warm_not_enough_messages`
- `test_should_warm_never_called_api`
- `test_reset_warm_counter`

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_group_timeline.py -v`
Expected: all remaining tests pass

- [ ] **Step 4: Run lint and type check**

Run: `uv run ruff check src/memory/group_timeline.py && uv run pyright src/memory/group_timeline.py`
Expected: clean

- [ ] **Step 5: Commit**

```bash
git add src/memory/group_timeline.py tests/test_group_timeline.py
git commit -m "refactor: remove warm-related fields from GroupTimeline"
```

---

### Task 3: PromptBuilder — inject identity.proactive into system blocks

**Files:**
- Modify: `src/llm/prompt.py`
- Modify: `tests/test_prompt.py`

- [ ] **Step 1: Read current test file**

Read `tests/test_prompt.py` to understand existing test patterns.

- [ ] **Step 2: Write the failing test**

Add to `tests/test_prompt.py`:

```python
@pytest.mark.asyncio
async def test_proactive_injected_in_group(long_term: LongTermMemory) -> None:
    """identity.proactive should be appended to system blocks in group chat."""
    identity = Identity(
        id="test", name="测试", personality="性格描述",
        proactive="【主动发言规则】积极参与群聊",
    )
    builder = PromptBuilder(long_term=long_term, instruction="")
    blocks = await builder.build_blocks(identity=identity, user_id="u1", group_id="g1")
    assert "【主动发言规则】积极参与群聊" in blocks[0]["text"]


@pytest.mark.asyncio
async def test_proactive_not_injected_without_group(long_term: LongTermMemory) -> None:
    """identity.proactive should NOT be injected for private chat."""
    identity = Identity(
        id="test", name="测试", personality="性格描述",
        proactive="【主动发言规则】积极参与群聊",
    )
    builder = PromptBuilder(long_term=long_term, instruction="")
    blocks = await builder.build_blocks(identity=identity, user_id="u1", group_id=None)
    assert "【主动发言规则】" not in blocks[0]["text"]


@pytest.mark.asyncio
async def test_proactive_not_injected_when_none(long_term: LongTermMemory) -> None:
    """No proactive rules when identity.proactive is None."""
    identity = Identity(
        id="test", name="测试", personality="性格描述",
        proactive=None,
    )
    builder = PromptBuilder(long_term=long_term, instruction="")
    blocks = await builder.build_blocks(identity=identity, user_id="u1", group_id="g1")
    assert "主动发言" not in blocks[0]["text"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompt.py::test_proactive_injected_in_group -v`
Expected: FAIL — proactive text not found in blocks

- [ ] **Step 4: Implement — add proactive injection to `build_blocks()`**

In `src/llm/prompt.py`, in `build_blocks()`, the method now receives `identity` which has a `proactive` field. Add the injection after the group_id line and before the `blocks.append(...)` call. The modified section of `build_blocks`:

```python
        if group_id:
            base_text += f"\n\n【当前在群 {group_id} 中对话】"
        if group_id and identity.proactive:
            base_text += "\n\n" + identity.proactive
        blocks.append({"type": "text", "text": base_text, "cache_control": {"type": "ephemeral"}})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/llm/prompt.py tests/test_prompt.py
git commit -m "feat: inject identity.proactive into system blocks for group chat"
```

---

### Task 4: LLMClient — add pass_turn tool, delete warm_cache and proactive_hint

**Files:**
- Modify: `src/llm/client.py`

- [ ] **Step 1: Add PASS_TURN_TOOL constant**

Add after the `_to_anthropic_tools` function (after line 76):

```python
_PASS_TURN_TOOL: dict[str, Any] = {
    "name": "pass_turn",
    "description": "当你认为不需要回复时调用此工具，跳过本轮发言。",
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "不回复的简短原因",
            }
        },
        "required": ["reason"],
    },
}
```

- [ ] **Step 2: Delete warm_cache from constructor**

Remove these parameters from `__init__`:
- `warm_enabled: bool = True`
- `warm_interval_messages: int = 10`
- `warm_ttl_seconds: int = 300`

Remove these instance variables from `__init__`:
- `self._warm_enabled = warm_enabled`
- `self._warm_interval = warm_interval_messages`
- `self._warm_ttl = warm_ttl_seconds`
- `self._warming = False`

- [ ] **Step 3: Delete `_warm_cache()` and `maybe_warm()` methods**

Delete the entire "缓存预热" section (lines 498-534): `_warm_cache()` and `maybe_warm()`.

- [ ] **Step 4: Modify `chat()` signature and pass_turn logic**

Change signature — remove `proactive_hint`, add `allow_skip`, return `str | None`:

```python
    async def chat(
        self,
        session_id: str,
        user_id: str,
        user_text: str,
        identity: Identity,
        group_id: str | None = None,
        ctx: ToolContext | None = None,
        on_segment: Callable[[str], Awaitable[None]] | None = None,
        allow_skip: bool = False,
    ) -> str | None:
```

Delete the `proactive_hint` injection block (lines 303-305):
```python
        # 主动插话：将回复对象提示作为对话消息注入，不污染 timeline
        if proactive_hint:
            messages.append({"role": "user", "content": proactive_hint})
```

Modify tool_defs construction to append pass_turn when `allow_skip=True`:

```python
        tool_defs: list[dict[str, Any]] | None = None
        if not self._tools.empty:
            tool_defs = _to_anthropic_tools(self._tools.to_openai_tools())
        if allow_skip:
            tool_defs = [*(tool_defs or []), _PASS_TURN_TOOL]
```

In the tool loop, add pass_turn interception right after `tool_uses` is extracted, before the existing `if not tool_uses:` check:

```python
            # Check for pass_turn
            pass_turn = next((tu for tu in tool_uses if tu.name == "pass_turn"), None)
            if pass_turn:
                reason = pass_turn.input.get("reason", "")
                elapsed = time.monotonic() - t0
                logger.info("pass_turn | session={} reason={!r} elapsed={:.1f}s", session_id, reason, elapsed)
                if is_group:
                    self._timeline.set_input_tokens(group_id, result["input_tokens"])
                return None
```

- [ ] **Step 5: Update return type for the tool-loop-exhausted path**

At the end of `chat()` (after `MAX_TOOL_ROUNDS` exhausted), the function already returns `last_seg` which is `str`. This is fine — exhaustion only happens when tools keep firing, and `pass_turn` exits early. No change needed.

- [ ] **Step 6: Run lint and type check**

Run: `uv run ruff check src/llm/client.py && uv run pyright src/llm/client.py`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add src/llm/client.py
git commit -m "feat: add pass_turn tool to LLMClient, delete warm_cache and proactive_hint"
```

---

### Task 5: GroupChatScheduler — create with tests

**Files:**
- Create: `src/llm/scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write tests first**

Create `tests/test_scheduler.py`:

```python
"""GroupChatScheduler unit tests."""

import asyncio

import pytest

from src.identity.models import Identity
from src.llm.scheduler import GroupChatScheduler
from src.memory.group_timeline import GroupTimeline


def _make_identity(proactive: str | None = "积极参与群聊") -> Identity:
    return Identity(id="test", name="测试", personality="测试人设", proactive=proactive)


class _FakeIdentityMgr:
    def __init__(self, identity: Identity) -> None:
        self._identity = identity

    def resolve(self) -> Identity:
        return self._identity


class _FakeLLM:
    """Records chat() calls and returns configured reply."""

    def __init__(self, reply: str | None = "你好") -> None:
        self.calls: list[dict] = []
        self.reply = reply

    async def chat(self, **kwargs) -> str | None:  # type: ignore[override]
        self.calls.append(kwargs)
        return self.reply


class TestNotify:
    @pytest.mark.asyncio
    async def test_no_proactive_skips(self) -> None:
        """notify is a no-op when identity.proactive is None."""
        identity = _make_identity(proactive=None)
        scheduler = GroupChatScheduler(
            llm=_FakeLLM(), timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(identity),  # type: ignore[arg-type]
            debounce_seconds=0.05, batch_size=100,
        )
        scheduler.notify("g1")
        assert "g1" not in scheduler._slots
        await scheduler.close()

    @pytest.mark.asyncio
    async def test_debounce_fires(self) -> None:
        """After debounce timeout, chat() is called."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=0.05, batch_size=100,
        )
        scheduler.notify("g1")
        await asyncio.sleep(0.15)
        assert len(llm.calls) == 1
        assert llm.calls[0]["allow_skip"] is True
        await scheduler.close()

    @pytest.mark.asyncio
    async def test_batch_size_fires_immediately(self) -> None:
        """Reaching batch_size triggers immediately without waiting for debounce."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=999, batch_size=3,
        )
        scheduler.notify("g1")
        scheduler.notify("g1")
        scheduler.notify("g1")
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()

    @pytest.mark.asyncio
    async def test_running_task_blocks_new_debounce(self) -> None:
        """While running_task is active, notify does not start new debounce."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=0.05, batch_size=100,
        )
        scheduler.notify("g1")
        await asyncio.sleep(0.15)  # debounce fires, running_task starts
        assert len(llm.calls) == 1
        scheduler.notify("g1")  # while running_task is active (or just finished)
        slot = scheduler._slots["g1"]
        # msg_count incremented but no new debounce if running_task is still set
        # (depends on timing, so just verify no crash)
        await scheduler.close()


class TestInterrupt:
    @pytest.mark.asyncio
    async def test_cancels_debounce(self) -> None:
        """interrupt cancels pending debounce task."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=999, batch_size=100,
        )
        scheduler.notify("g1")
        assert scheduler._slots["g1"].debounce_task is not None
        scheduler.interrupt("g1")
        assert scheduler._slots["g1"].debounce_task is None or scheduler._slots["g1"].debounce_task.cancelled()
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 0  # debounce was cancelled, no chat call
        await scheduler.close()

    @pytest.mark.asyncio
    async def test_interrupt_nonexistent_group(self) -> None:
        """interrupt on unknown group is a no-op."""
        scheduler = GroupChatScheduler(
            llm=_FakeLLM(), timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
        )
        scheduler.interrupt("unknown")  # should not raise
        await scheduler.close()


class TestClose:
    @pytest.mark.asyncio
    async def test_close_cancels_all(self) -> None:
        """close() cancels all pending tasks."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=999, batch_size=100,
        )
        scheduler.notify("g1")
        scheduler.notify("g2")
        await scheduler.close()
        # After close, debounce tasks should be cancelled
        for slot in scheduler._slots.values():
            assert slot.debounce_task is None or slot.debounce_task.cancelled()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: FAIL — `src.llm.scheduler` module not found

- [ ] **Step 3: Implement `src/llm/scheduler.py`**

```python
"""群聊统一调度器：debounce/batch 触发模型调用，@bot 抢占。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.memory.group_timeline import GroupTimeline
from src.tools.context import ToolContext

if TYPE_CHECKING:
    from nonebot.adapters.onebot.v11 import Bot, Message

    from src.identity import IdentityManager
    from src.llm.client import LLMClient


class _GroupSlot:
    __slots__ = ("debounce_task", "msg_count", "running_task")

    def __init__(self) -> None:
        self.debounce_task: asyncio.Task[None] | None = None
        self.running_task: asyncio.Task[None] | None = None
        self.msg_count: int = 0


class GroupChatScheduler:
    """群聊调度器：debounce 触发模型调用，@bot 抢占。"""

    def __init__(
        self,
        llm: LLMClient,
        timeline: GroupTimeline,
        identity_mgr: IdentityManager,
        debounce_seconds: float = 5.0,
        batch_size: int = 10,
    ) -> None:
        self._llm = llm
        self._timeline = timeline
        self._identity_mgr = identity_mgr
        self._debounce_seconds = debounce_seconds
        self._batch_size = batch_size
        self._slots: dict[str, _GroupSlot] = {}
        self._bot: Bot | None = None

    def set_bot(self, bot: Bot) -> None:
        self._bot = bot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify(self, group_id: str) -> None:
        """Called on every non-@ group message. Manages debounce/batch."""
        identity = self._identity_mgr.resolve()
        if identity.proactive is None:
            return

        slot = self._slots.setdefault(group_id, _GroupSlot())
        slot.msg_count += 1

        # If a chat() is already running, don't schedule another
        if slot.running_task and not slot.running_task.done():
            logger.debug("scheduler | group={} running, skip (msgs={})", group_id, slot.msg_count)
            return

        # Cancel old debounce
        if slot.debounce_task and not slot.debounce_task.done():
            slot.debounce_task.cancel()

        # Batch full → fire immediately
        if slot.msg_count >= self._batch_size:
            logger.info("scheduler | group={} batch full ({} msgs) → fire", group_id, slot.msg_count)
            self._fire(group_id)
        else:
            logger.debug("scheduler | group={} debounce start (msgs={})", group_id, slot.msg_count)
            slot.debounce_task = asyncio.create_task(self._debounce(group_id))

    def interrupt(self, group_id: str) -> None:
        """Called when @bot. Cancels debounce and running task for this group."""
        slot = self._slots.get(group_id)
        if not slot:
            return

        if slot.debounce_task and not slot.debounce_task.done():
            slot.debounce_task.cancel()
            slot.debounce_task = None
            logger.debug("scheduler | group={} debounce cancelled by @bot", group_id)

        if slot.running_task and not slot.running_task.done():
            slot.running_task.cancel()
            slot.running_task = None
            logger.info("scheduler | group={} running task cancelled by @bot", group_id)

        slot.msg_count = 0

    async def close(self) -> None:
        """Cancel all pending tasks on shutdown."""
        for slot in self._slots.values():
            for task in (slot.debounce_task, slot.running_task):
                if task and not task.done():
                    task.cancel()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _debounce(self, group_id: str) -> None:
        try:
            await asyncio.sleep(self._debounce_seconds)
            slot = self._slots.get(group_id)
            if slot and slot.msg_count > 0:
                logger.info("scheduler | group={} debounce fired ({} msgs)", group_id, slot.msg_count)
                self._fire(group_id)
        except asyncio.CancelledError:
            pass

    def _fire(self, group_id: str) -> None:
        slot = self._slots.get(group_id)
        if not slot:
            return
        slot.msg_count = 0
        slot.running_task = asyncio.create_task(self._do_chat(group_id))
        slot.running_task.add_done_callback(lambda _: None)

    async def _do_chat(self, group_id: str) -> None:
        slot = self._slots.get(group_id)
        try:
            if not self._bot:
                logger.warning("scheduler | group={} no bot set, skip", group_id)
                return

            identity = self._identity_mgr.resolve()
            session_id = f"group_{group_id}"
            ctx = ToolContext(bot=self._bot, user_id="", group_id=group_id)

            async def send_segment(text: str) -> None:
                if self._bot:
                    from nonebot.adapters.onebot.v11 import Message
                    await self._bot.send_group_msg(group_id=int(group_id), message=Message(text))

            reply = await self._llm.chat(
                session_id=session_id,
                user_id="",
                user_text="",
                identity=identity,
                group_id=group_id,
                ctx=ctx,
                on_segment=send_segment,
                allow_skip=True,
            )

            if reply and self._bot:
                from nonebot.adapters.onebot.v11 import Message
                await self._bot.send_group_msg(group_id=int(group_id), message=Message(reply))

        except asyncio.CancelledError:
            logger.debug("scheduler | group={} chat cancelled", group_id)
        except Exception:
            logger.exception("scheduler | group={} chat error", group_id)
        finally:
            if slot:
                slot.running_task = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: all pass

- [ ] **Step 5: Run lint and type check**

Run: `uv run ruff check src/llm/scheduler.py && uv run pyright src/llm/scheduler.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/llm/scheduler.py tests/test_scheduler.py
git commit -m "feat: add GroupChatScheduler with debounce, batch, and interrupt"
```

---

### Task 6: Plugin layer — wire up GroupChatScheduler, delete ProactiveEvaluator usage

**Files:**
- Modify: `src/plugins/chat/__init__.py`

- [ ] **Step 1: Replace imports**

Remove:
```python
from src.llm.proactive import ProactiveDecision, ProactiveEvaluator
```

Add:
```python
from src.llm.scheduler import GroupChatScheduler
```

- [ ] **Step 2: Replace global variables**

Change:
```python
_llm: LLMClient
_proactive: ProactiveEvaluator
_identity_mgr: IdentityManager
_timeline: GroupTimeline
_short_term: ShortTermMemory
```

To:
```python
_llm: LLMClient
_scheduler: GroupChatScheduler
_identity_mgr: IdentityManager
_timeline: GroupTimeline
_short_term: ShortTermMemory
```

- [ ] **Step 3: Update `_init()`**

Remove `warm_*` params from LLMClient constructor (lines 78-80).

Replace ProactiveEvaluator initialization (lines 83-94) with:

```python
    _scheduler = GroupChatScheduler(
        llm=_llm,
        timeline=_timeline,
        identity_mgr=_identity_mgr,
        debounce_seconds=bot_config.group.debounce_seconds,
        batch_size=bot_config.group.batch_size,
    )
```

Make sure `global` declaration includes `_scheduler` instead of `_proactive`.

- [ ] **Step 4: Update `_shutdown()`**

```python
@driver.on_shutdown
async def _shutdown() -> None:
    await _llm.close()
    await _scheduler.close()
```

- [ ] **Step 5: Update `_on_connect()`**

Add `_scheduler.set_bot(bot)` at the start. Delete the entire proactive history detection block (lines 126-167: the `identity = _identity_mgr.resolve()` loop through groups). The result:

```python
@driver.on_bot_connect
async def _on_connect(bot: Bot) -> None:
    """Bot 连接后拉取群历史消息，填充群聊上下文。"""
    _llm._bot_self_id = bot.self_id
    _scheduler.set_bot(bot)
    try:
        bot_config = load_config()
        group_list: list[dict[str, object]] = await bot.get_group_list()
        group_ids = [str(g["group_id"]) for g in group_list]
        if _allowed_groups:
            group_ids = [gid for gid in group_ids if int(gid) in _allowed_groups]
        logger.info("loading history | groups={}", len(group_ids))
        await load_group_history(
            napcat_url=bot_config.napcat.api_url,
            group_ids=group_ids,
            timeline=_timeline,
            count=bot_config.group.history_load_count,
            bot_self_id=bot.self_id,
        )
    except Exception:
        logger.exception("failed to load group history")
        return
    logger.info("Bot 就绪，开始接收消息 ✓")
```

- [ ] **Step 6: Simplify `collect_group_context()`**

Replace the `maybe_warm` + proactive callback block (lines 201-240) with a single line:

```python
    _scheduler.notify(group_id)
```

Full handler:

```python
@group_listener.handle()
async def collect_group_context(bot: Bot, event: GroupMessageEvent) -> None:
    if _allowed_groups and event.group_id not in _allowed_groups:
        return
    if str(event.user_id) == bot.self_id:
        return
    text = event.get_plaintext().strip()
    if not text:
        return

    nickname = event.sender.nickname or str(event.user_id)
    group_id = str(event.group_id)
    _timeline.add(
        group_id,
        role="user",
        speaker=f"{nickname}({event.user_id})",
        content=text,
    )

    if event.is_tome():
        return

    _scheduler.notify(group_id)
```

- [ ] **Step 7: Update `handle_chat()`**

Add `_scheduler.interrupt(group_id)` call before the chat, and remove `proactive_hint`:

```python
@chat.handle()
async def handle_chat(bot: Bot, event: MessageEvent) -> None:
    if isinstance(event, GroupMessageEvent):
        if _allowed_groups and event.group_id not in _allowed_groups:
            return
    else:
        if _allowed_private_users and event.user_id not in _allowed_private_users:
            return

    user_text = event.get_plaintext().strip()
    if not user_text:
        return

    sid = _session_id(event)
    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    identity = _identity_mgr.resolve()

    if group_id:
        _scheduler.interrupt(group_id)

    ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=group_id)

    async def send_segment(text: str) -> None:
        await bot.send(event, Message(text))

    try:
        reply = await _llm.chat(
            session_id=sid,
            user_id=str(event.user_id),
            user_text=user_text,
            identity=identity,
            group_id=group_id,
            ctx=ctx,
            on_segment=send_segment,
        )
    except Exception:
        logger.exception("chat error")
        reply = "出错了，请稍后再试"

    if reply:
        await chat.finish(Message(reply))
```

- [ ] **Step 8: Run lint and type check**

Run: `uv run ruff check src/plugins/chat/__init__.py && uv run pyright src/plugins/chat/__init__.py`
Expected: clean

- [ ] **Step 9: Commit**

```bash
git add src/plugins/chat/__init__.py
git commit -m "feat: wire GroupChatScheduler into plugin, remove ProactiveEvaluator usage"
```

---

### Task 7: Delete proactive.py and its tests

**Files:**
- Delete: `src/llm/proactive.py`
- Delete: `tests/test_proactive.py`

- [ ] **Step 1: Delete files**

```bash
git rm src/llm/proactive.py tests/test_proactive.py
```

- [ ] **Step 2: Verify no remaining imports**

Run: `uv run ruff check src/ && uv run pyright src/`
Expected: clean (no dangling imports to `proactive`)

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: delete ProactiveEvaluator and its tests"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full lint + type check + tests**

```bash
uv run ruff check src/ && uv run pyright && uv run pytest -v
```

Expected: all clean, all tests pass.

- [ ] **Step 2: Verify line count reduction**

```bash
git diff --stat HEAD~7
```

Expected: net deletion of ~280 lines.

- [ ] **Step 3: Commit any final fixes if needed**
