# Memo System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `.qmd` long-term memory with a markdown memo system: index + detail files, @/# cross-references, in-memory cache, new tools, updated compact prompts, new cache strategy, and a Dream agent.

**Architecture:** `MemoStore` (new) is the central store — file I/O, in-memory cache, lock management. `PromptBuilder` is refactored for the new 4-breakpoint cache layout. `RecallMemoTool` / `UpdateMemoTool` replace the old `SaveMemoryTool` / `RecallMemoryTool`. Compact prompts in `LLMClient` are updated to extract prose memos. Dream agent runs as a background LLM session.

**Tech Stack:** Python 3.12, aiofiles, asyncio, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-04-02-memo-system-design.md`

---

### File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/memory/memo_store.py` | `Memo` dataclass, `LockManager`, `MemoStore` (parse, cache, read, write, index) |
| Create | `src/tools/memo_tools.py` | `RecallMemoTool`, `UpdateMemoTool` |
| Create | `src/llm/dream.py` | Dream agent: trigger logic, pre-check, LLM session |
| Create | `tests/test_memo_store.py` | Unit tests for MemoStore |
| Create | `tests/test_memo_tools.py` | Unit tests for new tools |
| Create | `tests/test_dream.py` | Unit tests for Dream trigger/pre-check |
| Modify | `src/config.py` | Add `MemoConfig`, `CompactConfig`, `DreamConfig` |
| Modify | `src/llm/prompt.py` | Refactor `PromptBuilder` for static block + per-entity block |
| Modify | `src/llm/client.py` | New compact prompts, micro compact, circuit breaker, parallel tool exec, MemoStore integration |
| Modify | `src/plugins/chat/__init__.py` | Wire `MemoStore`, new tools, Dream scheduler, replace `LongTermMemory` |
| Modify | `tests/test_prompt.py` | Update for new `PromptBuilder` signature |
| Delete | `src/tools/memory_tool.py` | Replaced by `memo_tools.py` |
| Delete | `src/memory/long_term.py` | Replaced by `memo_store.py` |
| Delete | `tests/test_long_term.py` | Replaced by `test_memo_store.py` |

---

### Task 1: Config models

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Write test**

```python
# tests/test_config_loader.py — append to existing file

def test_memo_config_defaults() -> None:
    from src.config import MemoConfig, CompactConfig, DreamConfig
    m = MemoConfig()
    assert m.dir == "storage/memories"
    assert m.user_max_chars == 300
    assert m.group_max_chars == 500
    assert m.index_max_lines == 200
    assert m.history_enabled is True

    c = CompactConfig()
    assert c.micro_ratio == 0.6
    assert c.full_ratio == 0.8
    assert c.max_failures == 3

    d = DreamConfig()
    assert d.interval_hours == 24
    assert d.min_compacts == 5
    assert d.max_rounds == 15
```

- [ ] **Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_config_loader.py::test_memo_config_defaults -v`
Expected: `ImportError` — `MemoConfig` not defined yet.

- [ ] **Step 3: Implement**

Add to `src/config.py`:

```python
class MemoConfig(BaseModel):
    """备忘录系统配置。"""
    dir: str = "storage/memories"
    user_max_chars: int = 300
    group_max_chars: int = 500
    index_max_lines: int = 200
    history_enabled: bool = True

class CompactConfig(BaseModel):
    """上下文压缩配置。"""
    micro_ratio: float = 0.6
    full_ratio: float = 0.8
    max_failures: int = 3

class DreamConfig(BaseModel):
    """Dream 整理配置。"""
    interval_hours: int = 24
    min_compacts: int = 5
    max_rounds: int = 15
```

Update `ContextConfig` to remove `compact_ratio` (moved to `CompactConfig`). Update `BotConfig`:

```python
class BotConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    log: LogConfig = LogConfig()
    memo: MemoConfig = MemoConfig()
    compact: CompactConfig = CompactConfig()
    dream: DreamConfig = DreamConfig()
    soul: SoulConfig = SoulConfig()
    group: GroupConfig = GroupConfig()
    napcat: NapcatConfig = NapcatConfig()
    superusers: set[str] = set()
    allowed_private_users: list[int] = []
```

Keep old `memory: MemoryConfig` temporarily for backward compat — remove it in the final cleanup task.

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_config_loader.py -v`

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/config.py tests/test_config_loader.py --fix
git add src/config.py tests/test_config_loader.py
git commit -m "feat(config): add MemoConfig, CompactConfig, DreamConfig"
```

---

### Task 2: MemoStore core — parse, cache, read

**Files:**
- Create: `src/memory/memo_store.py`
- Create: `tests/test_memo_store.py`

- [ ] **Step 1: Write tests for Memo parsing and MemoStore read API**

```python
# tests/test_memo_store.py
import pytest
from src.memory.memo_store import Memo, MemoStore, parse_memo

SAMPLE_USER_MEMO = """\
<!-- updated: 2026-03-31 14:30 | source: compact:group:987654 -->

小明（明哥）｜杭州·后端·Go/Python

最近在学 Rust。和 @789012(小红) 在 #987654 互怼。
"""

SAMPLE_GROUP_MEMO = """\
<!-- updated: 2026-03-31 14:30 | source: compact:group:987654 -->

技术吹水群｜~10人｜轻松玩梗

@123456(小明) 和 @789012(小红) 经常互怼前后端。
"""


def test_parse_user_memo() -> None:
    memo = parse_memo("user_123456", SAMPLE_USER_MEMO)
    assert memo.id == "user_123456"
    assert memo.kind == "user"
    assert memo.identity == "小明（明哥）｜杭州·后端·Go/Python"
    assert memo.source == "compact:group:987654"
    assert "@789012" in memo.refs or "789012" in memo.refs
    assert "#987654" in memo.refs or "987654" in memo.refs


def test_parse_group_memo() -> None:
    memo = parse_memo("group_987654", SAMPLE_GROUP_MEMO)
    assert memo.kind == "group"
    assert "技术吹水群" in memo.identity


@pytest.fixture
def store(tmp_path: object) -> MemoStore:
    return MemoStore(base_dir=str(tmp_path))


async def test_store_empty_read(store: MemoStore) -> None:
    await store.startup()
    assert store.read("user_999") is None


async def test_store_write_and_read(store: MemoStore) -> None:
    await store.startup()
    await store.write("user_123456", SAMPLE_USER_MEMO.split("\n\n", 1)[1], "test")
    memo = store.read("user_123456")
    assert memo is not None
    assert memo.kind == "user"
    assert "小明" in memo.identity


async def test_store_mentions_index(store: MemoStore) -> None:
    await store.startup()
    await store.write("group_987654", SAMPLE_GROUP_MEMO.split("\n\n", 1)[1], "test")
    mentioned_in = store.about("123456")
    assert any("group_987654" in m.id for m in mentioned_in)


async def test_store_list_ids(store: MemoStore) -> None:
    await store.startup()
    await store.write("user_111", "测试｜test", "test")
    await store.write("group_222", "群｜test", "test")
    assert "user_111" in store.list_ids("user")
    assert "group_222" in store.list_ids("group")


async def test_store_serialize_index(store: MemoStore) -> None:
    await store.startup()
    await store.write("user_123456", "小明｜杭州后端\n\n和 @789012 互怼。", "test")
    index = store.serialize_index()
    assert "@123456" in index
    assert "小明" in index
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_memo_store.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/memory/memo_store.py`**

Implement: `Memo` dataclass, `parse_memo()` function, `LockManager`, `MemoStore` with `startup()`, `read()`, `write()`, `about()`, `list_ids()`, `serialize_index()`.

Key implementation details:
- `parse_memo(id, raw_text)`: regex for `<!-- updated: ... | source: ... -->`, first non-empty content line → identity, `re.findall(r"@(\d+)")` + `re.findall(r"#(\d+)")` → refs, `kind` derived from id prefix
- `write()`: acquire `_lock_mgr.get(id)`, create parent dirs, write `.md.tmp`, `os.rename`, append `.log`, re-parse into `_memos`, rebuild `_mentions`, rewrite `index.md`
- `startup()`: glob `users/*.md` + `groups/*.md`, parse each, build caches, clean `.tmp` residuals
- `_mentions` built by iterating all memos and collecting `refs` into reverse index
- `serialize_index()`: iterate `_memos`, output `# users\n` section + `# groups\n` section, one line per memo using `identity` + refs
- ID format: files named `{qq_num}.md`, internal ID is `user_{qq_num}` or `group_{group_id}`
- Path safety: reject IDs with `..` or `/`

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_memo_store.py -v`

- [ ] **Step 5: Lint + type check + commit**

```bash
uv run ruff check src/memory/memo_store.py tests/test_memo_store.py --fix
uv run pyright src/memory/memo_store.py
git add src/memory/memo_store.py tests/test_memo_store.py
git commit -m "feat(memory): add MemoStore with parse, cache, read/write, index"
```

---

### Task 3: MemoStore — write safety and changelog

**Files:**
- Modify: `src/memory/memo_store.py`
- Modify: `tests/test_memo_store.py`

- [ ] **Step 1: Write tests for atomic write and changelog**

```python
# tests/test_memo_store.py — append

import os

async def test_atomic_write_no_tmp_residual(store: MemoStore) -> None:
    """After write, no .tmp files should remain."""
    await store.startup()
    await store.write("user_100", "Test｜test", "test")
    tmp_files = list(store._base_dir.rglob("*.tmp"))
    assert len(tmp_files) == 0


async def test_changelog_appended(store: MemoStore, tmp_path: object) -> None:
    """Each write appends one line to .log."""
    await store.startup()
    await store.write("user_100", "V1｜first", "compact:group:999")
    await store.write("user_100", "V2｜second", "tool:abc")

    log_path = store._base_dir / "users" / "100.log"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert "compact:group:999" in lines[0]
    assert "tool:abc" in lines[1]


async def test_startup_cleans_tmp(store: MemoStore, tmp_path: object) -> None:
    """Startup removes leftover .tmp files."""
    users_dir = store._base_dir / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    (users_dir / "orphan.md.tmp").write_text("garbage")
    await store.startup()
    assert not (users_dir / "orphan.md.tmp").exists()


async def test_concurrent_writes_different_ids(store: MemoStore) -> None:
    """Writes to different IDs run in parallel without error."""
    import asyncio
    await store.startup()
    await asyncio.gather(
        store.write("user_1", "A｜a", "test"),
        store.write("user_2", "B｜b", "test"),
        store.write("group_3", "C｜c", "test"),
    )
    assert store.read("user_1") is not None
    assert store.read("user_2") is not None
    assert store.read("group_3") is not None
```

- [ ] **Step 2: Run tests, verify new tests fail (or pass if already implemented in Task 2)**

Run: `uv run pytest tests/test_memo_store.py -v`

- [ ] **Step 3: Ensure write safety and changelog are correct in implementation**

Verify `write()` in `memo_store.py`:
1. Creates parent dir (`users/` or `groups/`) if not exists
2. Writes metadata comment as first line: `<!-- updated: {now} | source: {source} -->`
3. Writes to `.md.tmp` then `os.rename` to `.md`
4. Appends one line to `.log` (create if needed): `{timestamp} | {source} | {summary}`
5. `startup()` globs and unlinks all `*.tmp`

- [ ] **Step 4: Run all tests, verify pass**

Run: `uv run pytest tests/test_memo_store.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/memory/memo_store.py tests/test_memo_store.py
git commit -m "feat(memory): atomic write safety and changelog"
```

---

### Task 4: Memo tools — RecallMemoTool and UpdateMemoTool

**Files:**
- Create: `src/tools/memo_tools.py`
- Create: `tests/test_memo_tools.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_memo_tools.py
import pytest
from src.memory.memo_store import MemoStore
from src.tools.memo_tools import RecallMemoTool, UpdateMemoTool
from src.tools.context import ToolContext


@pytest.fixture
async def store_with_data(tmp_path: object) -> MemoStore:
    s = MemoStore(base_dir=str(tmp_path))
    await s.startup()
    await s.write("user_123456", "小明（明哥）｜杭州·后端\n\n喜欢 Go。", "test")
    await s.write("user_789012", "小红｜前端·Rust\n\n和 @123456 互怼。", "test")
    await s.write("group_987654", "技术群｜10人\n\n@123456 @789012 活跃。", "test")
    return s


async def test_recall_by_id(store_with_data: MemoStore) -> None:
    tool = RecallMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456")
    result = await tool.execute(ctx, id="user_123456")
    assert "小明" in result
    assert "Go" in result


async def test_recall_by_query(store_with_data: MemoStore) -> None:
    tool = RecallMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456")
    result = await tool.execute(ctx, query="小红")
    assert "789012" in result  # QQ号 must appear for disambiguation


async def test_recall_not_found(store_with_data: MemoStore) -> None:
    tool = RecallMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456")
    result = await tool.execute(ctx, id="user_999999")
    assert "没有" in result or "未找到" in result


async def test_update_memo_returns_immediately(store_with_data: MemoStore) -> None:
    tool = UpdateMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456", session_id="sess_1")
    result = await tool.execute(ctx, id="user_123456", memo="小明｜新内容")
    assert "已提交" in result


async def test_update_memo_tool_schema() -> None:
    s = MemoStore(base_dir="/tmp/unused")
    tool = UpdateMemoTool(s)
    schema = tool.parameters
    assert "id" in schema["properties"]
    assert "memo" in schema["properties"]
    assert schema["required"] == ["id", "memo"]
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_memo_tools.py -v`

- [ ] **Step 3: Implement `src/tools/memo_tools.py`**

```python
"""Memo tools: recall and update user/group memos."""

import asyncio
from typing import Any

from src.memory.memo_store import MemoStore
from src.tools.base import Tool
from src.tools.context import ToolContext


class RecallMemoTool(Tool):
    def __init__(self, store: MemoStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "recall_memo"

    @property
    def description(self) -> str:
        return (
            "查找或读取用户/群的备忘录。"
            "用 id 精确读取完整内容，或用 query 按昵称/关键词搜索（返回摘要列表）。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "精确查找: user_QQ号 或 group_群号"},
                "query": {"type": "string", "description": "按昵称或关键词模糊搜索"},
                "kind": {"type": "string", "enum": ["user", "group"], "description": "过滤类型"},
            },
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        memo_id: str | None = kwargs.get("id")
        query: str | None = kwargs.get("query")
        kind: str | None = kwargs.get("kind")

        if memo_id:
            memo = self._store.read(memo_id)
            if memo is None:
                return f"未找到 {memo_id} 的备忘录。"
            return f"[{memo.id}]\n{memo.body}"

        if query:
            results: list[str] = []
            for mid in self._store.list_ids(kind):
                memo = self._store.read(mid)
                if memo and (query in memo.identity or query in memo.body):
                    results.append(f"- {memo.id}: {memo.identity}")
            if not results:
                return f"未找到与 '{query}' 相关的备忘录。"
            return "匹配结果：\n" + "\n".join(results)

        return "请提供 id 或 query 参数。"


class UpdateMemoTool(Tool):
    def __init__(self, store: MemoStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "update_memo"

    @property
    def description(self) -> str:
        return "更新用户或群的备忘录。传入完整新版内容（全文重写，不是追加）。异步执行，不阻塞对话。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "目标: user_QQ号 或 group_群号"},
                "memo": {"type": "string", "description": "完整的新版备忘录内容"},
            },
            "required": ["id", "memo"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        memo_id: str = kwargs["id"]
        memo_text: str = kwargs["memo"]
        session_id = getattr(ctx, "session_id", "unknown")
        asyncio.create_task(self._store.write(memo_id, memo_text, f"tool:{session_id}"))
        return "已提交更新"
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_memo_tools.py -v`

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/tools/memo_tools.py tests/test_memo_tools.py --fix
git add src/tools/memo_tools.py tests/test_memo_tools.py
git commit -m "feat(tools): add RecallMemoTool and UpdateMemoTool"
```

---

### Task 5: PromptBuilder refactor — static block + per-entity block

**Files:**
- Modify: `src/llm/prompt.py`
- Modify: `tests/test_prompt.py`

- [ ] **Step 1: Write tests for new PromptBuilder**

```python
# tests/test_prompt.py — rewrite
import pytest
from src.identity.models import Identity
from src.llm.prompt import PromptBuilder, load_instruction
from src.memory.memo_store import MemoStore


@pytest.fixture
async def store(tmp_path: object) -> MemoStore:
    s = MemoStore(base_dir=str(tmp_path))
    await s.startup()
    await s.write("user_100", "测试用户｜test", "test")
    await s.write("group_200", "测试群｜test", "test")
    return s


@pytest.fixture
def identity() -> Identity:
    return Identity(id="test", name="Bot", personality="I am a bot.", proactive="Proactive rules.")


def test_load_instruction_missing(tmp_path: object) -> None:
    assert load_instruction(str(tmp_path)) == ""


def test_load_instruction_exists(tmp_path: object) -> None:
    (tmp_path / "instruction.md").write_text("Do things.")
    assert load_instruction(str(tmp_path)) == "Do things."


async def test_build_static_called_once(identity: Identity) -> None:
    pb = PromptBuilder(instruction="Test instruction.")
    pb.build_static(identity, bot_self_id="999")
    assert pb._static_block is not None
    assert "I am a bot." in pb._static_block["text"]
    assert "Test instruction." in pb._static_block["text"]
    assert "Proactive rules." in pb._static_block["text"]
    assert pb._static_block["cache_control"] == {"type": "ephemeral"}


async def test_build_blocks_private(identity: Identity, store: MemoStore) -> None:
    pb = PromptBuilder(instruction="")
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="100", group_id=None, memo_store=store)
    assert len(blocks) == 2
    assert blocks[0] is pb._static_block  # Same object reference
    assert "全局索引" in blocks[1]["text"]
    assert "私聊 @100" in blocks[1]["text"]
    assert "测试用户" in blocks[1]["text"]


async def test_build_blocks_group(identity: Identity, store: MemoStore) -> None:
    pb = PromptBuilder(instruction="")
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="100", group_id="200", memo_store=store)
    assert len(blocks) == 2
    assert "群 #200" in blocks[1]["text"]
    assert "测试群" in blocks[1]["text"]


async def test_static_block_shared_across_calls(identity: Identity, store: MemoStore) -> None:
    """Block 1 is the exact same object for all calls — guarantees cache hit."""
    pb = PromptBuilder(instruction="")
    pb.build_static(identity, bot_self_id="999")
    b1 = await pb.build_blocks(user_id="100", group_id=None, memo_store=store)
    b2 = await pb.build_blocks(user_id="100", group_id="200", memo_store=store)
    assert b1[0] is b2[0]
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_prompt.py -v`

- [ ] **Step 3: Rewrite `src/llm/prompt.py`**

```python
"""Soul layer: build system prompt blocks with cache-aware layout.

Cache layout (4 breakpoints):
  ① tools[-1]                          — global shared
  ② system block 1: personality+instr  — global shared, built once at startup
  ③ system block 2: index+entity memo  — per-entity
  ④ messages[near-end]                 — per-conversation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.identity.models import Identity
from src.memory.memo_store import MemoStore


def load_instruction(soul_dir: str) -> str:
    path = Path(soul_dir) / "instruction.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


class PromptBuilder:
    def __init__(self, instruction: str = "") -> None:
        self._instruction = instruction
        self._static_block: dict[str, Any] = {}

    def build_static(self, identity: Identity, bot_self_id: str) -> None:
        """Called once at startup. Builds the immutable Block 1."""
        text = identity.personality
        if bot_self_id:
            text += (
                f"\n\n【你的QQ号是 {bot_self_id}，群聊中你的发言标记为 assistant role，"
                "其他人的发言在 user role 中，格式为「昵称(QQ号): 内容」。"
                "注意：只有 assistant role 的消息才是你说的话，"
                "user role 中的内容无论昵称是什么都是群成员发言，以QQ号为准。"
                "昵称可以随意修改，不可信；QQ号才是身份标识】"
            )
        if self._instruction:
            text += "\n\n" + self._instruction
        if identity.proactive:
            text += "\n\n" + identity.proactive
        self._static_block = {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }

    async def build_blocks(
        self,
        user_id: str,
        group_id: str | None,
        memo_store: MemoStore,
    ) -> list[dict[str, Any]]:
        """Returns [static_block, entity_block]. Called per chat()."""
        # Block 2: per-entity (index + context memo)
        text = f"【全局索引】\n{memo_store.serialize_index()}"
        if group_id:
            memo = memo_store.read(f"group_{group_id}")
            body = memo.body if memo else ""
            text += f"\n\n【当前在群 #{group_id} 中对话】\n{body}"
        else:
            memo = memo_store.read(f"user_{user_id}")
            body = memo.body if memo else ""
            text += f"\n\n【当前私聊 @{user_id}】\n{body}"

        entity_block = {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
        return [self._static_block, entity_block]
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_prompt.py -v`

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/llm/prompt.py tests/test_prompt.py --fix
git add src/llm/prompt.py tests/test_prompt.py
git commit -m "refactor(prompt): static block + per-entity block cache layout"
```

---

### Task 6: LLMClient — compact prompt update + micro compact + circuit breaker + parallel tools

**Files:**
- Modify: `src/llm/client.py`

- [ ] **Step 1: Update `__init__` to accept `MemoStore` instead of `LongTermMemory`**

Replace `long_term: LongTermMemory | None = None` with `memo_store: MemoStore | None = None`. Store as `self._memo_store`. Remove `self._long_term`.

Update `_compact_group` signature: replace LongTermMemory writes with `self._memo_store.write()`.

Update `build_blocks` call: pass `memo_store` instead of `user_id` lookup.

- [ ] **Step 2: Update `_compact_group` prompt to request prose memo rewrites**

Replace the current JSON schema (traits/event extraction) with:

```python
system = [{"type": "text", "text": (
    "你是一个对话分析助手。请完成两个任务：\n"
    "1. 将以下群聊记录压缩成简洁的中文摘要。保留关键信息。\n"
    "2. 更新相关用户和群的备忘录（全文重写，不是追加）。\n\n"
    f"当前用户备忘录：\n{user_memos_context}\n\n"
    f"当前群备忘录：\n{group_memo_context}\n\n"
    "以 JSON 输出：\n"
    '{"summary": "摘要", "memos": {"QQ号": "完整新版用户备忘录", ...}, '
    '"group_memo": "完整新版群备忘录"}\n\n'
    "备忘录规则：用 @QQ号 标注人物，#群号 标注群。QQ号是唯一身份标识，昵称不可信。\n"
    "记印象和结论，不记流水账。没有新信息的用户可省略。只输出 JSON。"
)}]
```

After parsing, batch all writes:

```python
if new_summary:
    self._timeline.compact(group_id, split, new_summary)
if self._memo_store:
    for uid, memo_text in memos.items():
        await self._memo_store.write(f"user_{uid}", memo_text, f"compact:group:{group_id}")
    if group_memo:
        await self._memo_store.write(f"group_{group_id}", group_memo, f"compact:group:{group_id}")
```

- [ ] **Step 3: Update `_compact` (private) to also extract memos**

Similar to group compact but simpler — only one user. Add current user memo as context, request updated memo in JSON output.

- [ ] **Step 4: Add micro compact**

Add `_micro_compact_group` and `_micro_compact_private` methods:

```python
def _micro_compact_group(self, group_id: str) -> None:
    """Drop oldest 25% of messages. No LLM call, no summary change."""
    assert self._timeline is not None
    messages = self._timeline.get_messages(group_id)
    if len(messages) < 4:
        return
    drop = len(messages) // 4
    self._timeline.drop_oldest(group_id, drop)
    logger.info("micro_compact_group | group={} dropped={}", group_id, drop)
```

Update the compact decision in `chat()`:

```python
ratio = input_tokens / self._max_context_tokens
if ratio > self._full_ratio:
    await self._compact_group(group_id, identity)
elif ratio > self._micro_ratio:
    self._micro_compact_group(group_id)
```

- [ ] **Step 5: Add circuit breaker**

```python
_compact_failures: int = 0

async def _compact_group(self, group_id, identity):
    if self._compact_failures >= self._max_compact_failures:
        logger.warning("compact circuit breaker active | group={}", group_id)
        return
    try:
        # ... existing compact logic ...
        self._compact_failures = 0
    except Exception:
        self._compact_failures += 1
        logger.exception("compact failed ({}/{})", self._compact_failures, self._max_compact_failures)
```

- [ ] **Step 6: Add parallel tool execution**

Replace the sequential loop in `chat()`:

```python
# Before (sequential):
for tu in tool_uses:
    tool_result = await self._tools.call(tu.name, json.dumps(tu.input), ctx=tool_ctx)
    ...

# After (parallel):
call_results = await asyncio.gather(*[
    self._tools.call(tu.name, json.dumps(tu.input), ctx=tool_ctx)
    for tu in tool_uses
])
tool_results = []
for tu, result_text in zip(tool_uses, call_results):
    logger.debug("tool_result | name={} result={!r}", tu.name, result_text[:200])
    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_text})
```

- [ ] **Step 7: Update `build_blocks` call**

```python
# Before:
system_blocks = await self._prompt.build_blocks(
    identity=identity, user_id=user_id, group_id=group_id, bot_self_id=self._bot_self_id,
)

# After:
system_blocks = await self._prompt.build_blocks(
    user_id=user_id, group_id=group_id, memo_store=self._memo_store,
)
```

- [ ] **Step 8: Run full test suite**

Run: `uv run pytest -v`
Fix any import errors from the PromptBuilder signature change.

- [ ] **Step 9: Commit**

```bash
git add src/llm/client.py
git commit -m "refactor(client): memo store integration, micro compact, circuit breaker, parallel tools"
```

---

### Task 7: GroupTimeline — add `drop_oldest` method

**Files:**
- Modify: `src/memory/group_timeline.py`
- Modify: `tests/test_group_timeline.py`

- [ ] **Step 1: Write test**

```python
# tests/test_group_timeline.py — append

async def test_drop_oldest(timeline: GroupTimeline) -> None:
    for i in range(10):
        timeline.add("g1", role="user", speaker=f"u{i}", content=f"msg{i}")
    assert len(timeline.get_messages("g1")) == 10
    timeline.drop_oldest("g1", 3)
    messages = timeline.get_messages("g1")
    assert len(messages) == 7
    assert messages[0]["content"] == "msg3"
```

- [ ] **Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_group_timeline.py::test_drop_oldest -v`

- [ ] **Step 3: Implement**

Add to `GroupTimeline`:

```python
def drop_oldest(self, group_id: str, count: int) -> None:
    """Drop the oldest `count` messages. For micro compact."""
    state = self._groups.get(group_id)
    if state is None:
        return
    state.messages = state.messages[count:]
```

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_group_timeline.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/memory/group_timeline.py tests/test_group_timeline.py
git commit -m "feat(timeline): add drop_oldest for micro compact"
```

---

### Task 8: Dream agent

**Files:**
- Create: `src/llm/dream.py`
- Create: `tests/test_dream.py`

- [ ] **Step 1: Write tests for trigger logic and pre-check**

```python
# tests/test_dream.py
import pytest
from unittest.mock import AsyncMock
from src.llm.dream import DreamAgent, dream_pre_check
from src.memory.memo_store import MemoStore


@pytest.fixture
async def store(tmp_path: object) -> MemoStore:
    s = MemoStore(base_dir=str(tmp_path))
    await s.startup()
    await s.write("user_100", "用户A｜test\n\n提到 @999(不存在)。", "test")
    await s.write("group_200", "群B｜test\n\n@100 活跃。", "test")
    return s


def test_pre_check_finds_dangling_refs(store: MemoStore) -> None:
    issues = dream_pre_check(store)
    assert any("999" in issue for issue in issues)


def test_pre_check_no_issues_when_clean(tmp_path: object) -> None:
    s = MemoStore(base_dir=str(tmp_path))
    import asyncio
    asyncio.get_event_loop().run_until_complete(s.startup())
    asyncio.get_event_loop().run_until_complete(
        s.write("user_100", "A｜test", "test")
    )
    issues = dream_pre_check(s)
    assert len(issues) == 0


async def test_dream_agent_should_run(store: MemoStore) -> None:
    agent = DreamAgent(store=store, interval_hours=0, min_compacts=0, max_rounds=5)
    agent._last_dream_time = 0
    agent._compacts_since_dream = 1
    assert agent.should_run()


async def test_dream_agent_not_yet(store: MemoStore) -> None:
    agent = DreamAgent(store=store, interval_hours=24, min_compacts=5, max_rounds=5)
    assert not agent.should_run()
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_dream.py -v`

- [ ] **Step 3: Implement `src/llm/dream.py`**

```python
"""Dream agent: periodic background memory consolidation."""

import asyncio
import time
from typing import Any

from loguru import logger

from src.memory.memo_store import MemoStore


def dream_pre_check(store: MemoStore) -> list[str]:
    """Programmatic scan for structural issues. Returns list of issue descriptions."""
    issues: list[str] = []
    all_ids = set(store.list_ids())
    for mid in all_ids:
        memo = store.read(mid)
        if memo is None:
            continue
        for ref in memo.refs:
            ref_id = f"user_{ref}" if not ref.startswith("group_") else ref
            # Check @user refs
            if ref_id.startswith("user_") and ref_id not in all_ids:
                issues.append(f"{mid}: @{ref} has no memo")
        # Check oversized
        if memo.kind == "user" and len(memo.body) > 500:
            issues.append(f"{mid}: user memo oversized ({len(memo.body)} chars)")
        if memo.kind == "group" and len(memo.body) > 800:
            issues.append(f"{mid}: group memo oversized ({len(memo.body)} chars)")
    return issues


class DreamAgent:
    def __init__(
        self,
        store: MemoStore,
        interval_hours: int = 24,
        min_compacts: int = 5,
        max_rounds: int = 15,
    ) -> None:
        self._store = store
        self._interval_hours = interval_hours
        self._min_compacts = min_compacts
        self._max_rounds = max_rounds
        self._last_dream_time: float = time.time()
        self._compacts_since_dream: int = 0
        self._running: bool = False

    def notify_compact(self) -> None:
        """Called after each successful compact."""
        self._compacts_since_dream += 1

    def should_run(self) -> bool:
        if self._running:
            return False
        elapsed_hours = (time.time() - self._last_dream_time) / 3600
        return (
            elapsed_hours >= self._interval_hours
            and self._compacts_since_dream >= self._min_compacts
        )

    async def maybe_run(self, llm_call: Any) -> None:
        """Check trigger and run if conditions met. Fire-and-forget."""
        if self.should_run():
            asyncio.create_task(self._run(llm_call))

    async def _run(self, llm_call: Any) -> None:
        if self._running:
            return
        self._running = True
        try:
            logger.info("dream starting")
            issues = dream_pre_check(self._store)
            # Build system prompt with index + issues
            index_text = self._store.serialize_index()
            issues_text = "\n".join(f"- {i}" for i in issues) if issues else "无明显问题"
            system = [{
                "type": "text",
                "text": (
                    "你是记忆整理助手。以下是当前索引：\n"
                    f"{index_text}\n\n"
                    f"预检发现以下问题：\n{issues_text}\n\n"
                    "请自主检查备忘录质量并修复问题。"
                    f"限制：最多 {self._max_rounds} 轮工具调用。"
                ),
            }]
            # Run agent loop with recall_memo + update_memo tools
            # Uses the same LLM call infrastructure as main chat
            await llm_call(system)
            self._last_dream_time = time.time()
            self._compacts_since_dream = 0
            logger.info("dream completed")
        except Exception:
            logger.exception("dream failed")
        finally:
            self._running = False
```

Note: The actual `llm_call` integration with tools will be wired in Task 9 when integrating into the plugin.

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_dream.py -v`

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/llm/dream.py tests/test_dream.py --fix
git add src/llm/dream.py tests/test_dream.py
git commit -m "feat(dream): Dream agent with trigger logic and pre-check"
```

---

### Task 9: Wire everything in plugin init

**Files:**
- Modify: `src/plugins/chat/__init__.py`

- [ ] **Step 1: Replace LongTermMemory with MemoStore**

```python
# Remove:
from src.memory.long_term import LongTermMemory
from src.tools.memory_tool import RecallMemoryTool, SaveMemoryTool

# Add:
from src.memory.memo_store import MemoStore
from src.tools.memo_tools import RecallMemoTool, UpdateMemoTool
from src.llm.dream import DreamAgent
```

- [ ] **Step 2: Update `_init()` function**

```python
@driver.on_startup
async def _init() -> None:
    global _llm, _scheduler, _identity_mgr, _timeline, _short_term, _allowed_groups, _allowed_private_users, _dream

    bot_config = load_config()
    # ...

    memo_store = MemoStore(base_dir=bot_config.memo.dir)
    await memo_store.startup()

    _short_term = ShortTermMemory()
    _timeline = GroupTimeline(max_messages=bot_config.group.max_timeline_messages)
    instruction = load_instruction(bot_config.soul.dir)
    prompt_builder = PromptBuilder(instruction=instruction)

    # ... identity loading ...

    prompt_builder.build_static(identity_mgr.resolve(), bot_self_id="")  # bot_self_id set on_connect

    tools = ToolRegistry()
    tools.register(RecallMemoTool(memo_store))
    tools.register(UpdateMemoTool(memo_store))
    tools.register(DateTimeTool())
    tools.register(WebFetchTool())
    tools.register(HttpApiTool())
    tools.register(MuteUserTool(superusers))
    tools.register(SetTitleTool(superusers))
    tools.register(SendGroupMsgTool(superusers))

    _llm = LLMClient(
        base_url=bot_config.llm.base_url,
        api_key=bot_config.llm.api_key,
        model=bot_config.llm.model,
        prompt_builder=prompt_builder,
        short_term=_short_term,
        tools=tools,
        max_context_tokens=bot_config.llm.context.max_context_tokens,
        micro_ratio=bot_config.compact.micro_ratio,
        full_ratio=bot_config.compact.full_ratio,
        max_compact_failures=bot_config.compact.max_failures,
        group_timeline=_timeline,
        memo_store=memo_store,
    )

    _dream = DreamAgent(
        store=memo_store,
        interval_hours=bot_config.dream.interval_hours,
        min_compacts=bot_config.dream.min_compacts,
        max_rounds=bot_config.dream.max_rounds,
    )

    # ... scheduler ...
```

- [ ] **Step 3: Update `_on_connect` to call `build_static` with real bot_self_id**

```python
@driver.on_bot_connect
async def _on_connect(bot: Bot) -> None:
    _llm._bot_self_id = bot.self_id
    _llm._prompt.build_static(_identity_mgr.resolve(), bot_self_id=bot.self_id)
    # ... rest unchanged ...
```

- [ ] **Step 4: Add ToolContext.session_id field**

In `src/tools/context.py`, add `session_id: str = ""` to `ToolContext` dataclass. Pass it from `handle_chat`:

```python
ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=group_id, session_id=sid)
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Run: `uv run ruff check src/ --fix`
Run: `uv run pyright`

- [ ] **Step 6: Commit**

```bash
git add src/plugins/chat/__init__.py src/tools/context.py
git commit -m "refactor(plugin): wire MemoStore, new tools, Dream agent"
```

---

### Task 10: Add instruction chapter + cleanup old files

**Files:**
- Modify: `soul/instruction.md`
- Delete: `src/memory/long_term.py`
- Delete: `src/tools/memory_tool.py`
- Delete: `tests/test_long_term.py`

- [ ] **Step 1: Add memo writing rules to instruction.md**

Append to `soul/instruction.md`:

```markdown
## 备忘录书写规范

你可以使用 recall_memo 和 update_memo 工具来读取和更新用户/群的备忘录。

### 身份识别
QQ号是唯一身份标识。昵称不可信——可以随意修改、可以重名、可能被故意冒用。

### 标记约定
- 用 @QQ号 标注人物，首次提及带昵称：@123456(小明)
- 用 #群号 标注群：#987654
- 永远不要只写昵称不写QQ号

### 写作规则
1. 记印象和结论，不记流水账
2. 个人信息写用户备忘录，群体事件写群备忘录
3. 用户备忘录 300 字以内，群备忘录 500 字以内
4. 只记自述或明确观察到的事实，不做推测
5. 不记你自己说了什么
6. 昵称冲突时以QQ号为准

### 更新方式
调用 update_memo 时传入完整新版内容（全文重写，不是追加）。合并新旧信息，保留重要的，丢弃过时的。
```

- [ ] **Step 2: Delete old files**

```bash
rm src/memory/long_term.py
rm src/tools/memory_tool.py
rm tests/test_long_term.py
```

- [ ] **Step 3: Remove old MemoryConfig from config.py if still present**

Clean up any backward-compat shim left from Task 1.

- [ ] **Step 4: Run full test suite + lint + type check**

```bash
uv run pytest -v
uv run ruff check src/ tests/ --fix
uv run pyright
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(instruction): add memo writing rules; remove old qmd memory system"
```

---

### Task 11: Final integration test

**Files:**
- Modify: `tests/test_memo_store.py`

- [ ] **Step 1: Write an end-to-end test**

```python
# tests/test_memo_store.py — append

async def test_full_lifecycle(tmp_path: object) -> None:
    """Startup → write users + group → read → mentions → index → overwrite → verify."""
    store = MemoStore(base_dir=str(tmp_path))
    await store.startup()

    # Create
    await store.write("user_100", "小明｜杭州后端\n\n和 @200(小红) 在 #300 互怼。", "compact:group:300")
    await store.write("user_200", "小红｜前端\n\n和 @100(小明) 互怼。", "compact:group:300")
    await store.write("group_300", "技术群｜5人\n\n@100 @200 活跃。", "compact:group:300")

    # Read
    assert store.read("user_100") is not None
    assert store.read("user_100").identity == "小明｜杭州后端"

    # Mentions
    mentioned_in = store.about("100")
    ids = {m.id for m in mentioned_in}
    assert "user_200" in ids      # 小红's memo mentions @100
    assert "group_300" in ids     # group memo mentions @100

    # Index
    index = store.serialize_index()
    assert "@100" in index
    assert "@200" in index
    assert "#300" in index

    # Overwrite (full rewrite)
    await store.write("user_100", "小明（老明）｜杭州后端·转Go\n\n换了工作。", "tool:sess1")
    memo = store.read("user_100")
    assert "老明" in memo.identity
    assert "转Go" in memo.identity

    # Changelog
    log_path = store._base_dir / "users" / "100.log"
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2

    # Restart and verify persistence
    store2 = MemoStore(base_dir=str(tmp_path))
    await store2.startup()
    assert store2.read("user_100") is not None
    assert "老明" in store2.read("user_100").identity
    assert len(store2.list_ids("user")) == 2
    assert len(store2.list_ids("group")) == 1
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_memo_store.py::test_full_lifecycle -v`

- [ ] **Step 3: Run full suite**

```bash
uv run pytest -v
uv run ruff check src/ tests/
uv run pyright
```

- [ ] **Step 4: Final commit**

```bash
git add tests/test_memo_store.py
git commit -m "test: add memo system integration test"
```
