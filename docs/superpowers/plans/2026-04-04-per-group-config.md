# Per-Group Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow each monitored group to have its own configuration (blocked users, @-only mode, debounce/batch/history overrides) with global defaults as fallback.

**Architecture:** Add `GroupOverride` and `ResolvedGroupConfig` models to `src/config.py`. `GroupConfig.resolve(group_id)` merges global defaults with per-group overrides. The scheduler and listener consume resolved configs per group instead of global scalars.

**Tech Stack:** Pydantic v2, TOML (tomllib), asyncio

---

### Task 1: Config models and resolve logic

**Files:**
- Modify: `src/config.py:44-51` (GroupConfig) — add new models and fields
- Test: `tests/test_config_loader.py` — add resolve tests

- [ ] **Step 1: Write failing tests for resolve()**

Add to `tests/test_config_loader.py`:

```python
from src.config import GroupConfig, GroupOverride, ResolvedGroupConfig


class TestGroupConfigResolve:
    def test_resolve_no_override(self) -> None:
        """No override for group — returns global defaults."""
        cfg = GroupConfig(
            debounce_seconds=5.0, batch_size=10, at_only=False,
            blocked_users=[100], history_load_count=30,
        )
        resolved = cfg.resolve(999)
        assert resolved.blocked_users == {100}
        assert resolved.at_only is False
        assert resolved.debounce_seconds == 5.0
        assert resolved.batch_size == 10
        assert resolved.history_load_count == 30

    def test_resolve_full_override(self) -> None:
        """Override supplies all fields — all override values win."""
        cfg = GroupConfig(
            debounce_seconds=5.0, batch_size=10, blocked_users=[100],
            overrides={
                123: GroupOverride(
                    blocked_users=[200], at_only=True,
                    debounce_seconds=10.0, batch_size=20, history_load_count=50,
                ),
            },
        )
        resolved = cfg.resolve(123)
        assert resolved.blocked_users == {100, 200}
        assert resolved.at_only is True
        assert resolved.debounce_seconds == 10.0
        assert resolved.batch_size == 20
        assert resolved.history_load_count == 50

    def test_resolve_partial_override_falls_back(self) -> None:
        """Override only sets at_only — rest falls back to global."""
        cfg = GroupConfig(
            debounce_seconds=5.0, batch_size=10,
            overrides={123: GroupOverride(at_only=True)},
        )
        resolved = cfg.resolve(123)
        assert resolved.at_only is True
        assert resolved.debounce_seconds == 5.0
        assert resolved.batch_size == 10
        assert resolved.history_load_count == 30

    def test_resolve_blocked_users_union(self) -> None:
        """blocked_users is the union of global and per-group lists."""
        cfg = GroupConfig(
            blocked_users=[1, 2],
            overrides={123: GroupOverride(blocked_users=[2, 3])},
        )
        resolved = cfg.resolve(123)
        assert resolved.blocked_users == {1, 2, 3}

    def test_resolve_override_at_only_false_overrides_global_true(self) -> None:
        """Per-group at_only=False overrides global at_only=True."""
        cfg = GroupConfig(
            at_only=True,
            overrides={123: GroupOverride(at_only=False)},
        )
        resolved = cfg.resolve(123)
        assert resolved.at_only is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_loader.py::TestGroupConfigResolve -v`
Expected: ImportError — `GroupOverride`, `ResolvedGroupConfig` don't exist yet.

- [ ] **Step 3: Implement config models**

In `src/config.py`, replace the existing `GroupConfig` (lines 44-51) with:

```python
class ResolvedGroupConfig(BaseModel):
    """resolve() 返回的扁平群配置，所有字段已合并。"""

    blocked_users: set[int] = set()
    at_only: bool = False
    debounce_seconds: float = 5.0
    batch_size: int = 10
    history_load_count: int = 30


class GroupOverride(BaseModel):
    """单个群的覆盖配置，None 表示使用全局值。"""

    blocked_users: list[int] = []
    at_only: bool | None = None
    debounce_seconds: float | None = None
    batch_size: int | None = None
    history_load_count: int | None = None


class GroupConfig(BaseModel):
    """群聊上下文配置。"""

    history_load_count: int = 30
    allowed_groups: list[int] = []
    debounce_seconds: float = 5.0
    batch_size: int = 10
    at_only: bool = False
    blocked_users: list[int] = []
    overrides: dict[int, GroupOverride] = {}

    def resolve(self, group_id: int) -> ResolvedGroupConfig:
        """合并全局默认值与单群覆盖，返回最终生效配置。"""
        base_blocked = set(self.blocked_users)
        override = self.overrides.get(group_id)
        if override is None:
            return ResolvedGroupConfig(
                blocked_users=base_blocked,
                at_only=self.at_only,
                debounce_seconds=self.debounce_seconds,
                batch_size=self.batch_size,
                history_load_count=self.history_load_count,
            )
        return ResolvedGroupConfig(
            blocked_users=base_blocked | set(override.blocked_users),
            at_only=override.at_only if override.at_only is not None else self.at_only,
            debounce_seconds=override.debounce_seconds if override.debounce_seconds is not None else self.debounce_seconds,
            batch_size=override.batch_size if override.batch_size is not None else self.batch_size,
            history_load_count=override.history_load_count if override.history_load_count is not None else self.history_load_count,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_loader.py::TestGroupConfigResolve -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Add TOML parsing test for overrides**

Add to `tests/test_config_loader.py`:

```python
def test_group_overrides_from_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TOML [group.overrides.<id>] sections parse into GroupConfig.overrides."""
    monkeypatch.delenv("BOT_CONFIG_PATH", raising=False)
    toml_file = tmp_path / "config.toml"
    _write_toml(
        toml_file,
        """
[group]
blocked_users = [100]
at_only = false

[group.overrides.100001]
blocked_users = [200, 300]
at_only = true
debounce_seconds = 10.0

[group.overrides.100002]
batch_size = 20
history_load_count = 50
""",
    )
    cfg = load_config(config_path=str(toml_file))

    assert cfg.group.blocked_users == [100]
    assert cfg.group.at_only is False

    assert 100001 in cfg.group.overrides
    o1 = cfg.group.overrides[100001]
    assert o1.blocked_users == [200, 300]
    assert o1.at_only is True
    assert o1.debounce_seconds == 10.0
    assert o1.batch_size is None

    assert 100002 in cfg.group.overrides
    o2 = cfg.group.overrides[100002]
    assert o2.batch_size == 20
    assert o2.history_load_count == 50
    assert o2.at_only is None
```

- [ ] **Step 6: Run the TOML test**

Run: `uv run pytest tests/test_config_loader.py::test_group_overrides_from_toml -v`
Expected: PASS.

- [ ] **Step 7: Run full test suite to check nothing broke**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: All existing + new tests PASS. The new `at_only` and `blocked_users` fields on `GroupConfig` have defaults that match old behavior.

- [ ] **Step 8: Commit**

```bash
git add src/config.py tests/test_config_loader.py
git commit -m "feat: add per-group config models with resolve() method"
```

---

### Task 2: Scheduler per-group config support

**Files:**
- Modify: `src/llm/scheduler.py:31-48` (constructor), `57-91` (notify), `120-128` (_debounce)
- Modify: `tests/test_scheduler.py` — update constructor calls, add at_only tests

- [ ] **Step 1: Write failing tests for at_only and per-group params**

Add to `tests/test_scheduler.py`:

```python
from src.config import GroupConfig, GroupOverride


class TestAtOnly:
    async def test_at_only_skips_debounce(self) -> None:
        """at_only=True: non-@ messages don't trigger debounce or batch."""
        llm = _FakeLLM(reply=None)
        group_config = GroupConfig(at_only=True, debounce_seconds=0.05, batch_size=3)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=group_config,
        )
        scheduler.notify("123")
        scheduler.notify("123")
        scheduler.notify("123")  # reaches batch_size but at_only blocks it
        await asyncio.sleep(0.15)
        assert len(llm.calls) == 0
        await scheduler.close()

    async def test_at_only_still_fires_on_at(self) -> None:
        """at_only=True: @ messages still fire immediately."""
        llm = _FakeLLM(reply=None)
        group_config = GroupConfig(at_only=True, debounce_seconds=999, batch_size=100)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=group_config,
        )
        scheduler.notify("123", is_at=True)
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_per_group_at_only_override(self) -> None:
        """Group 123 is at_only, group 456 is not."""
        llm = _FakeLLM(reply=None)
        group_config = GroupConfig(
            at_only=False, debounce_seconds=0.05, batch_size=100,
            overrides={123: GroupOverride(at_only=True)},
        )
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=group_config,
        )
        scheduler.notify("123")  # at_only group — no debounce
        scheduler.notify("456")  # normal group — debounce starts
        await asyncio.sleep(0.15)
        assert len(llm.calls) == 1  # only group 456 fired
        assert llm.calls[0]["session_id"] == "group_456"
        await scheduler.close()


class TestPerGroupParams:
    async def test_per_group_debounce(self) -> None:
        """Group 123 has 0.3s debounce (override), group 456 uses global 0.05s."""
        llm = _FakeLLM(reply=None)
        group_config = GroupConfig(
            debounce_seconds=0.05, batch_size=100,
            overrides={123: GroupOverride(debounce_seconds=0.3)},
        )
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=group_config,
        )
        scheduler.notify("123")
        scheduler.notify("456")
        await asyncio.sleep(0.15)
        # group 456 (0.05s debounce) should have fired, group 123 (0.3s) not yet
        assert len(llm.calls) == 1
        assert llm.calls[0]["session_id"] == "group_456"
        await asyncio.sleep(0.3)
        assert len(llm.calls) == 2
        await scheduler.close()

    async def test_per_group_batch_size(self) -> None:
        """Group 123 has batch_size=2 (override), global is 100."""
        llm = _FakeLLM(reply=None)
        group_config = GroupConfig(
            debounce_seconds=999, batch_size=100,
            overrides={123: GroupOverride(batch_size=2)},
        )
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=group_config,
        )
        scheduler.notify("123")
        scheduler.notify("123")  # hits batch_size=2
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_scheduler.py::TestAtOnly -v`
Expected: TypeError — `GroupChatScheduler.__init__()` doesn't accept `group_config`.

- [ ] **Step 3: Refactor scheduler constructor**

Replace the constructor in `src/llm/scheduler.py` (lines 34-48):

```python
    def __init__(
        self,
        llm: LLMClient,
        timeline: GroupTimeline,
        identity_mgr: IdentityManager,
        group_config: GroupConfig,
    ) -> None:
        self._llm = llm
        self._timeline = timeline
        self._identity_mgr = identity_mgr
        self._group_config = group_config
        self._slots: dict[str, _GroupSlot] = {}
        self._bot: Bot | None = None
```

Add runtime import at top of file (after the existing imports, **not** inside `TYPE_CHECKING`):

```python
from src.config import GroupConfig
```

- [ ] **Step 4: Refactor notify() for per-group config**

Replace `notify()` in `src/llm/scheduler.py` (lines 57-91):

```python
    def notify(self, group_id: str, *, is_at: bool = False) -> None:
        """Called on every group message. Manages debounce/batch."""
        identity = self._identity_mgr.resolve()
        if identity.proactive is None:
            return

        resolved = self._group_config.resolve(int(group_id))

        slot = self._slots.setdefault(group_id, _GroupSlot())
        slot.msg_count += 1

        if is_at:
            if slot.running_task and not slot.running_task.done():
                slot.pending_at = True
                logger.debug("scheduler | group={} @ queued (task running)", group_id)
                return
            if slot.debounce_task and not slot.debounce_task.done():
                slot.debounce_task.cancel()
            logger.info("scheduler | group={} @ -> fire", group_id)
            self._fire(group_id)
            return

        # at_only mode: only respond to @ messages
        if resolved.at_only:
            logger.debug("scheduler | group={} at_only, skip (msgs={})", group_id, slot.msg_count)
            return

        if slot.running_task and not slot.running_task.done():
            logger.debug("scheduler | group={} busy, skip (msgs={})", group_id, slot.msg_count)
            return

        if slot.debounce_task and not slot.debounce_task.done():
            slot.debounce_task.cancel()

        if slot.msg_count >= resolved.batch_size:
            logger.info("scheduler | group={} batch full ({} msgs) -> fire", group_id, slot.msg_count)
            self._fire(group_id)
        else:
            logger.debug("scheduler | group={} debounce start (msgs={})", group_id, slot.msg_count)
            slot.debounce_task = asyncio.create_task(
                self._debounce(group_id, resolved.debounce_seconds)
            )
```

- [ ] **Step 5: Update _debounce() to accept per-group seconds**

Replace `_debounce()` in `src/llm/scheduler.py` (lines 120-128):

```python
    async def _debounce(self, group_id: str, seconds: float) -> None:
        try:
            await asyncio.sleep(seconds)
            slot = self._slots.get(group_id)
            if slot and slot.msg_count > 0:
                logger.info("scheduler | group={} debounce fired ({} msgs)", group_id, slot.msg_count)
                self._fire(group_id)
        except asyncio.CancelledError:
            pass
```

- [ ] **Step 6: Update existing tests to use group_config**

In `tests/test_scheduler.py`, update every `GroupChatScheduler(...)` call in `TestNotify`, `TestAtHandling`, and `TestClose` to use the new constructor signature. Replace `debounce_seconds=X, batch_size=Y` with `group_config=GroupConfig(debounce_seconds=X, batch_size=Y)`.

Add import at top:

```python
from src.config import GroupConfig, GroupOverride
```

Example — `TestNotify.test_no_proactive_skips`:

```python
    async def test_no_proactive_skips(self) -> None:
        identity = _make_identity(proactive=None)
        scheduler = GroupChatScheduler(
            llm=_FakeLLM(), timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(identity),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.notify("g1")
        assert "g1" not in scheduler._slots
        await scheduler.close()
```

Apply the same pattern to all other test methods in `TestNotify`, `TestAtHandling`, and `TestClose`.

- [ ] **Step 7: Run all scheduler tests**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: All existing + new tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/llm/scheduler.py tests/test_scheduler.py
git commit -m "feat: scheduler uses per-group config for at_only, debounce, batch"
```

---

### Task 3: History loader per-group count

**Files:**
- Modify: `src/memory/history_loader.py:20-37` — add `counts` parameter

- [ ] **Step 1: Add optional counts parameter to load_group_history**

In `src/memory/history_loader.py`, update the signature (lines 20-28) and loop (lines 31-37):

```python
async def load_group_history(
    napcat_url: str,
    group_ids: list[str],
    timeline: GroupTimeline,
    count: int = 30,
    bot_self_id: str = "",
    image_cache: ImageCache | None = None,
    sticker_store: StickerStore | None = None,
    counts: dict[str, int] | None = None,
) -> None:
    """从 NapCat 拉取多个群的历史消息。"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        for gid in group_ids:
            gid_count = counts.get(gid, count) if counts else count
            try:
                await _load_one_group(
                    session, napcat_url, gid, timeline, gid_count, bot_self_id, image_cache, sticker_store
                )
            except Exception:
                logger.warning("load_history failed | group={}", gid, exc_info=True)
```

- [ ] **Step 2: Run existing history tests to verify no regression**

Run: `uv run pytest tests/test_history_sticker.py tests/test_history_self_messages.py -v`
Expected: PASS — the new `counts` param defaults to `None`, preserving old behavior.

- [ ] **Step 3: Commit**

```bash
git add src/memory/history_loader.py
git commit -m "feat: history loader accepts per-group message counts"
```

---

### Task 4: Plugin wiring — blocked users and per-group config

**Files:**
- Modify: `src/plugins/chat/__init__.py:53` (add global), `62-68` (init), `172-178` (scheduler), `217-231` (history), `349-376` (listener)

- [ ] **Step 1: Add `_group_config` global and initialize it**

In `src/plugins/chat/__init__.py`, add a new global (near line 53):

```python
from src.config import GroupConfig
```

Add to the global declarations (after `_allowed_groups`):

```python
_group_config: GroupConfig = GroupConfig()
```

In `_init()` (after `_allowed_groups = set(bot_config.group.allowed_groups)`, line 68):

```python
    global _group_config
    _group_config = bot_config.group
```

(Add `_group_config` to the existing `global` line 64.)

- [ ] **Step 2: Update scheduler initialization**

Replace the scheduler initialization (lines 172-178):

```python
    _scheduler = GroupChatScheduler(
        llm=_llm,
        timeline=_timeline,
        identity_mgr=_identity_mgr,
        group_config=bot_config.group,
    )
```

- [ ] **Step 3: Add blocked_users filter to group listener**

In `collect_group_context` (after the bot self_id check at line 354), add:

```python
    # Check per-group blocked users
    resolved = _group_config.resolve(event.group_id)
    if event.user_id in resolved.blocked_users:
        return
```

- [ ] **Step 4: Wire per-group history_load_count in _on_connect**

Replace the history loading block in `_on_connect` (lines 222-231):

```python
        logger.info("loading history | groups={}", len(group_ids))
        counts = {gid: _group_config.resolve(int(gid)).history_load_count for gid in group_ids}
        await load_group_history(
            napcat_url=bot_config.napcat.api_url,
            group_ids=group_ids,
            timeline=_timeline,
            count=bot_config.group.history_load_count,
            bot_self_id=bot.self_id,
            image_cache=_image_cache if _vision_enabled else None,
            sticker_store=_sticker_store,
            counts=counts,
        )
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS.

- [ ] **Step 6: Run lint and type check**

Run: `uv run ruff check src/ && uv run pyright`
Expected: No errors.

- [ ] **Step 7: Commit**

```bash
git add src/plugins/chat/__init__.py
git commit -m "feat: wire per-group blocked_users, at_only, and history count"
```

---

### Task 5: Update config.example.toml

**Files:**
- Modify: `config.example.toml:110-122`

- [ ] **Step 1: Update the [group] section**

Replace the `[group]` section in `config.example.toml` (lines 110-122):

```toml
# ---------------------------------------------------------------------------
# 群聊
# ---------------------------------------------------------------------------
[group]
# 启动时从 NapCat 拉取的历史消息条数
history_load_count = 30

# 群聊白名单，只处理这些群的消息。空数组 = 不限制
# allowed_groups = [100001, 100002]
allowed_groups = []

# 非@消息的 debounce 等待时间（秒），等消息间隙后触发模型调用
debounce_seconds = 5.0

# 最多攒多少条消息强制触发模型调用
batch_size = 10

# 仅@模式：开启后只在被@时回复，不主动插话
at_only = false

# 全局屏蔽用户列表（所有群生效）
blocked_users = []

# ---------------------------------------------------------------------------
# 群聊单群覆盖（可选）
# 未填的参数自动使用上方全局值，blocked_users 取并集
# ---------------------------------------------------------------------------
# [group.overrides.100001]
# blocked_users = [123456, 789012]
# at_only = true
# debounce_seconds = 10.0
# batch_size = 20
# history_load_count = 50
```

- [ ] **Step 2: Commit**

```bash
git add config.example.toml
git commit -m "docs: add per-group override examples to config.example.toml"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run full test suite, lint, and type check**

```bash
uv run pytest -v && uv run ruff check src/ && uv run pyright
```

Expected: All pass, no errors.
