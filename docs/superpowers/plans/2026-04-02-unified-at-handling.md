# Unified @ Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify @bot and non-@ message handling so the bot autonomously decides whether to reply in all scenarios, with a single scheduler queue.

**Architecture:** Remove the dedicated `chat` handler for group @bot messages. All group messages flow through `group_listener` → unified scheduler. The scheduler gains an `is_at` flag: @ messages fire immediately (or queue as `pending_at` if busy), non-@ messages use existing debounce/batch. The `allow_skip` parameter is deleted from `LLMClient.chat()` — `pass_turn` is always honored.

**Tech Stack:** Python 3.12, NoneBot2, pytest + pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-02-unified-at-handling-design.md`

---

### Task 1: Update Scheduler — Remove interrupt/release, Add `is_at` + `pending_at`

**Files:**
- Modify: `src/llm/scheduler.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Update `_FakeLLM` to remove `allow_skip` from tests**

In `tests/test_scheduler.py`, the existing `test_debounce_fires` asserts `llm.calls[0]["allow_skip"] is True`. This assertion must be removed since `allow_skip` is going away. Update the test:

```python
# In class TestNotify:
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
        await scheduler.close()
```

- [ ] **Step 2: Replace `TestInterrupt` with `TestAtHandling` tests**

Delete the entire `TestInterrupt` class. Replace with new tests for the `is_at` and `pending_at` behavior:

```python
class TestAtHandling:
    async def test_at_fires_immediately(self) -> None:
        """notify(is_at=True) fires immediately, skipping debounce."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=999, batch_size=100,
        )
        scheduler.notify("g1", is_at=True)
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_at_cancels_pending_debounce(self) -> None:
        """notify(is_at=True) cancels a pending debounce and fires immediately."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=999, batch_size=100,
        )
        scheduler.notify("g1")  # starts debounce
        assert scheduler._slots["g1"].debounce_task is not None
        scheduler.notify("g1", is_at=True)  # cancels debounce, fires immediately
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_at_queues_when_busy(self) -> None:
        """notify(is_at=True) sets pending_at when a task is already running."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=0.05, batch_size=100,
        )
        scheduler.notify("g1")  # debounce
        await asyncio.sleep(0.15)  # fires, running_task active
        assert len(llm.calls) == 1
        scheduler.notify("g1", is_at=True)  # should queue
        assert scheduler._slots["g1"].pending_at is True
        await scheduler.close()

    async def test_pending_at_fires_after_completion(self) -> None:
        """After running task completes, pending_at triggers a new call."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=0.05, batch_size=100,
        )
        scheduler.notify("g1")
        await asyncio.sleep(0.15)  # first call fires
        assert len(llm.calls) == 1
        scheduler.notify("g1", is_at=True)  # queued as pending_at
        await asyncio.sleep(0.2)  # first call finishes, pending fires
        assert len(llm.calls) == 2
        assert scheduler._slots["g1"].pending_at is False
        await scheduler.close()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: FAIL — `notify()` doesn't accept `is_at` yet, `pending_at` doesn't exist.

- [ ] **Step 4: Implement scheduler changes**

In `src/llm/scheduler.py`:

a) Update `_GroupSlot` — replace `interrupted` with `pending_at`:

```python
class _GroupSlot:
    __slots__ = ("debounce_task", "msg_count", "pending_at", "running_task")

    def __init__(self) -> None:
        self.debounce_task: asyncio.Task[None] | None = None
        self.running_task: asyncio.Task[None] | None = None
        self.msg_count: int = 0
        self.pending_at: bool = False
```

b) Update `notify()` — add `is_at` parameter, remove `interrupted` check:

```python
def notify(self, group_id: str, *, is_at: bool = False) -> None:
    """Called on every group message. Manages debounce/batch."""
    identity = self._identity_mgr.resolve()
    if identity.proactive is None:
        return

    slot = self._slots.setdefault(group_id, _GroupSlot())
    slot.msg_count += 1

    if is_at:
        if slot.running_task and not slot.running_task.done():
            slot.pending_at = True
            logger.debug("scheduler | group={} @ queued (task running)", group_id)
            return
        # Cancel pending debounce and fire immediately
        if slot.debounce_task and not slot.debounce_task.done():
            slot.debounce_task.cancel()
        logger.info("scheduler | group={} @ -> fire", group_id)
        self._fire(group_id)
        return

    # Non-@ path: debounce / batch (unchanged)
    if slot.running_task and not slot.running_task.done():
        logger.debug("scheduler | group={} busy, skip (msgs={})", group_id, slot.msg_count)
        return

    if slot.debounce_task and not slot.debounce_task.done():
        slot.debounce_task.cancel()

    if slot.msg_count >= self._batch_size:
        logger.info("scheduler | group={} batch full ({} msgs) -> fire", group_id, slot.msg_count)
        self._fire(group_id)
    else:
        logger.debug("scheduler | group={} debounce start (msgs={})", group_id, slot.msg_count)
        slot.debounce_task = asyncio.create_task(self._debounce(group_id))
```

c) Delete `interrupt()` and `release()` methods entirely.

d) Update `_do_chat()` — remove `allow_skip`, add `pending_at` drain in finally:

```python
async def _do_chat(self, group_id: str) -> None:
    slot = self._slots.get(group_id)
    try:
        identity = self._identity_mgr.resolve()
        session_id = f"group_{group_id}"
        ctx = ToolContext(bot=self._bot, user_id="", group_id=group_id)

        async def on_segment(text: str) -> None:
            await self._send_to_group(group_id, text)

        reply = await self._llm.chat(
            session_id=session_id,
            user_id="",
            user_text="",
            identity=identity,
            group_id=group_id,
            ctx=ctx,
            on_segment=on_segment if self._bot else None,
        )

        if reply:
            await self._send_to_group(group_id, reply)

    except asyncio.CancelledError:
        logger.debug("scheduler | group={} chat cancelled", group_id)
    except Exception:
        logger.exception("scheduler | group={} chat error", group_id)
    finally:
        if slot:
            slot.running_task = None
            if slot.pending_at:
                slot.pending_at = False
                self._fire(group_id)
```

e) Update docstrings: class docstring to `"""群聊统一调度器：debounce/batch/@触发模型调用。"""`, module docstring to `"""群聊统一调度器：debounce/batch/@ 触发模型调用，统一队列。"""`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: All PASS.

- [ ] **Step 6: Run lint and type check**

Run: `uv run ruff check src/llm/scheduler.py tests/test_scheduler.py && uv run pyright src/llm/scheduler.py`
Expected: Clean.

- [ ] **Step 7: Commit**

```bash
git add src/llm/scheduler.py tests/test_scheduler.py
git commit -m "refactor: unify scheduler — remove interrupt/release, add is_at + pending_at"
```

---

### Task 2: Update LLMClient — Remove `allow_skip`

**Files:**
- Modify: `src/llm/client.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write a test for pass_turn always being honored**

There are no existing tests for the pass_turn / allow_skip behavior in `test_client.py`. Add one to verify the new behavior (pass_turn always returns None):

```python
class TestPassTurn:
    async def test_pass_turn_returns_none(self, prompt, short_term, tools, timeline, memo_store) -> None:
        """pass_turn is always honored — chat() returns None."""
        async for client in _client(prompt, short_term, tools, timeline=timeline, memo_store=memo_store):
            gid = "12345"
            timeline.add(gid, role="user", content="hello", speaker="user(111)")

            mock_result = {
                "text": "",
                "tool_uses": [_ToolUse(id="tu_1", name="pass_turn", input={"reason": "not relevant"})],
                "input_tokens": 100,
            }
            with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_result):
                result = await client.chat(
                    session_id="group_12345",
                    user_id="111",
                    user_text="hello",
                    identity=_IDENTITY,
                    group_id=gid,
                    ctx=None,
                )
            assert result is None
```

This requires importing `_ToolUse` from `src.llm.client`. Add to the imports at the top of the test file:

```python
from src.llm.client import LLMClient, _ToolUse
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_client.py::TestPassTurn::test_pass_turn_returns_none -v`
Expected: FAIL — `allow_skip` defaults to False, so pass_turn is currently filtered out instead of honored.

- [ ] **Step 3: Update `chat()` — remove `allow_skip`, simplify pass_turn logic**

In `src/llm/client.py`:

a) Remove `allow_skip` from the `chat()` signature (line 350):

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
    ) -> str | None:
```

b) Replace the pass_turn branching logic (lines 412-423):

```python
            # Check for pass_turn
            pass_turn = next((tu for tu in tool_uses if tu.name == "pass_turn"), None)
            if pass_turn:
                reason = pass_turn.input.get("reason", "")
                elapsed = time.monotonic() - t0
                logger.info("pass_turn | session={} reason={!r} elapsed={:.1f}s", session_id, reason, elapsed)
                if is_group and group_id is not None and self._timeline is not None:
                    self._timeline.set_input_tokens(group_id, result["input_tokens"])
                return None
```

c) Update the comment on line 396:

```python
        # pass_turn is always available
        tool_defs = [*(tool_defs or []), _PASS_TURN_TOOL]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_client.py -v`
Expected: All PASS.

- [ ] **Step 5: Run lint and type check**

Run: `uv run ruff check src/llm/client.py tests/test_client.py && uv run pyright src/llm/client.py`
Expected: Clean.

- [ ] **Step 6: Commit**

```bash
git add src/llm/client.py tests/test_client.py
git commit -m "refactor: remove allow_skip — pass_turn is always honored"
```

---

### Task 3: Update Plugin Layer — Unify Group Handler, Add Private Handler

**Files:**
- Modify: `src/plugins/chat/__init__.py`

- [ ] **Step 1: Modify `collect_group_context()` to pass `is_at` to scheduler**

Replace lines 214-217. Currently:

```python
    if event.is_tome():
        return

    _scheduler.notify(group_id)
```

New logic:

```python
    _scheduler.notify(group_id, is_at=event.is_tome())
```

- [ ] **Step 2: Delete the `chat` handler and refactor `handle_chat` into a private-only handler**

Delete lines 222-279 (the `chat` matcher and `handle_chat` function). Replace with a private-message-only handler:

```python
# ── 私聊 ──

private_chat = on_message(rule=to_me(), priority=10, block=True)


@private_chat.handle()
async def handle_private_chat(bot: Bot, event: MessageEvent) -> None:
    if isinstance(event, GroupMessageEvent):
        return
    if _allowed_private_users and event.user_id not in _allowed_private_users:
        return

    reply_msg = getattr(event, "reply", None)
    user_text = _render_message(event.get_message(), reply=reply_msg)
    if not user_text:
        return

    sid = _session_id(event)
    identity = _identity_mgr.resolve()
    ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=None, session_id=sid)

    async def send_segment(text: str) -> None:
        await bot.send(event, Message(text))

    try:
        reply = await _llm.chat(
            session_id=sid,
            user_id=str(event.user_id),
            user_text=user_text,
            identity=identity,
            group_id=None,
            ctx=ctx,
            on_segment=send_segment,
        )
    except Exception:
        logger.exception("chat error")
        reply = "出错了，请稍后再试"

    if _dream_enabled:
        await _dream.maybe_run(_dream_llm_call)

    if reply:
        await private_chat.finish(Message(reply))
```

- [ ] **Step 3: Remove the `interrupt`/`release` calls from `_init` imports and `_on_connect`**

The `_on_connect` function at line 150-152 calls `_scheduler.trigger(gid)` — this stays as-is (trigger is unchanged).

No other code in the plugin calls `interrupt()` or `release()`, so no further cleanup is needed.

- [ ] **Step 4: Clean up unused import**

The `to_me` import is still needed for `private_chat`. No import changes needed.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All PASS.

- [ ] **Step 6: Run lint and type check**

Run: `uv run ruff check src/plugins/chat/__init__.py && uv run pyright src/plugins/chat/__init__.py`
Expected: Clean.

- [ ] **Step 7: Commit**

```bash
git add src/plugins/chat/__init__.py
git commit -m "refactor: unify group handler, move private chat to standalone handler"
```

---

### Task 4: Update Soul Instructions

**Files:**
- Modify: `soul/identity.md`

- [ ] **Step 1: Update the proactive section**

Replace lines 24-28 of `soul/identity.md`:

```markdown
只有以下情况才插话：
- 有人 @ 你或明确在跟你说话
- 有人明确向群里求助，且你确实能提供有价值的回答
- 话题和你的专业（物理、科学）直接相关，且你有独特的见解可以补充
- 有人聊到你，且你回应能让对话更有趣或更有价值
```

With:

```markdown
只有以下情况才插话：
- 有人明确在跟你说话或向你提问
- 有人明确向群里求助，且你确实能提供有价值的回答
- 话题和你的专业（物理、科学）直接相关，且你有独特的见解可以补充
- 有人聊到你，且你回应能让对话更有趣或更有价值

有人 @ 你表示对方在跟你说话，但这不意味着你必须回复——同样按以上标准自行判断。
```

- [ ] **Step 2: Commit**

```bash
git add soul/identity.md
git commit -m "docs: update proactive rules — @ no longer guarantees reply"
```

---

### Task 5: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All PASS.

- [ ] **Step 2: Run lint on all changed files**

Run: `uv run ruff check src/plugins/chat/__init__.py src/llm/scheduler.py src/llm/client.py`
Expected: Clean.

- [ ] **Step 3: Run type check**

Run: `uv run pyright`
Expected: Clean (or no new errors).

- [ ] **Step 4: Grep for stale references**

Verify no remaining references to `allow_skip`, `interrupt`, or `release` in source code:

```bash
grep -rn "allow_skip\|\.interrupt(\|\.release(" src/
```

Expected: No matches.
