# 对话历史 Compact 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将对话历史从 deque FIFO 截断改为无限累积 + 按需 LLM 压缩，最大化 Anthropic prompt cache 命中率。

**Architecture:** ShortTermMemory 改用 list 累积消息 + 存储摘要。LLMClient 在每次请求后记录 API 返回的 input_tokens，下次请求前检查是否需要 compact。Compact 时用同一模型将历史前半部分压缩成摘要。群聊上下文仅在有摘要时注入 messages。

**Tech Stack:** Python 3.12, aiohttp, pytest, Anthropic Messages API

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/config.py` | Modify | 新增 `llm_max_context_tokens`, `compact_ratio`；去掉 `short_term_max_rounds` |
| `src/memory/short_term.py` | Rewrite | deque → list，新增 summary/token 追踪/compact 方法 |
| `tests/test_short_term.py` | Rewrite | 适配新 API，新增 compact 相关测试 |
| `tests/conftest.py` | Modify | `short_term` fixture 去掉 `max_rounds` 参数 |
| `src/llm/client.py` | Modify | `_call_api` 返回 input_tokens；`chat()` 改消息拼装逻辑；新增 `_compact()` |
| `src/plugins/chat/__init__.py` | Modify | 适配新的构造函数签名 |

---

### Task 1: 重写 ShortTermMemory

**Files:**
- Modify: `src/memory/short_term.py`
- Modify: `tests/test_short_term.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: 更新 conftest fixture**

```python
# tests/conftest.py — 修改 short_term fixture
@pytest.fixture
def short_term() -> ShortTermMemory:
    return ShortTermMemory()
```

- [ ] **Step 2: 写 test_add_and_get 测试（适配新 API）**

```python
# tests/test_short_term.py — 完整重写
from src.memory.short_term import ShortTermMemory


def test_add_and_get(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "你好")
    short_term.add("s1", "assistant", "你好呀")
    msgs = short_term.get("s1")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "你好呀"


def test_session_isolation(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "消息1")
    short_term.add("s2", "user", "消息2")
    assert len(short_term.get("s1")) == 1
    assert len(short_term.get("s2")) == 1
    assert short_term.get("s1")[0]["content"] == "消息1"


def test_clear(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "hello")
    short_term.clear("s1")
    assert short_term.get("s1") == []


def test_get_empty(short_term: ShortTermMemory) -> None:
    assert short_term.get("nonexistent") == []
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_short_term.py -v`
Expected: FAIL — `ShortTermMemory()` 仍需要 `max_rounds` 参数

- [ ] **Step 4: 重写 short_term.py 基础功能（add/get/clear）**

```python
# src/memory/short_term.py
"""短期记忆：每个会话累积对话历史，按需 compact。"""

from typing import Literal, TypedDict

_MAX_SESSIONS = 500


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class _SessionState:
    __slots__ = ("messages", "summary", "last_input_tokens")

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []
        self.summary: str = ""
        self.last_input_tokens: int = 0


class ShortTermMemory:
    def __init__(self) -> None:
        self._store: dict[str, _SessionState] = {}

    def _get_or_create(self, session_id: str) -> _SessionState:
        if session_id not in self._store:
            if len(self._store) >= _MAX_SESSIONS:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[session_id] = _SessionState()
        return self._store[session_id]

    def add(self, session_id: str, role: Literal["user", "assistant"], content: str) -> None:
        state = self._get_or_create(session_id)
        state.messages.append(ChatMessage(role=role, content=content))

    def get(self, session_id: str) -> list[ChatMessage]:
        if session_id not in self._store:
            return []
        return list(self._store[session_id].messages)

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def get_summary(self, session_id: str) -> str:
        if session_id not in self._store:
            return ""
        return self._store[session_id].summary

    def set_input_tokens(self, session_id: str, tokens: int) -> None:
        if session_id in self._store:
            self._store[session_id].last_input_tokens = tokens

    def get_input_tokens(self, session_id: str) -> int:
        if session_id not in self._store:
            return 0
        return self._store[session_id].last_input_tokens

    def needs_compact(self, session_id: str, max_tokens: int, ratio: float) -> bool:
        return self.get_input_tokens(session_id) > max_tokens * ratio

    def compact(self, session_id: str, split: int, new_summary: str) -> None:
        if session_id not in self._store:
            return
        state = self._store[session_id]
        state.messages = state.messages[split:]
        state.summary = new_summary
        state.last_input_tokens = 0
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_short_term.py -v`
Expected: 4 PASSED

- [ ] **Step 6: 写 summary/token/compact 测试**

在 `tests/test_short_term.py` 末尾追加：

```python
def test_summary_empty_by_default(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "hello")
    assert short_term.get_summary("s1") == ""
    assert short_term.get_summary("nonexistent") == ""


def test_input_tokens_tracking(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "hello")
    assert short_term.get_input_tokens("s1") == 0
    short_term.set_input_tokens("s1", 5000)
    assert short_term.get_input_tokens("s1") == 5000
    # 不存在的 session 不报错
    assert short_term.get_input_tokens("nonexistent") == 0


def test_needs_compact(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "hello")
    short_term.set_input_tokens("s1", 150_000)
    assert short_term.needs_compact("s1", max_tokens=200_000, ratio=0.7)  # 150k > 140k
    assert not short_term.needs_compact("s1", max_tokens=200_000, ratio=0.8)  # 150k < 160k
    assert not short_term.needs_compact("nonexistent", max_tokens=200_000, ratio=0.7)


def test_compact(short_term: ShortTermMemory) -> None:
    for i in range(10):
        short_term.add("s1", "user", f"u{i}")
        short_term.add("s1", "assistant", f"a{i}")
    short_term.set_input_tokens("s1", 99999)

    # compact 前半（10条消息）
    short_term.compact("s1", split=10, new_summary="对话摘要：用户打了10次招呼")
    msgs = short_term.get("s1")
    assert len(msgs) == 10
    assert msgs[0]["content"] == "u5"
    assert short_term.get_summary("s1") == "对话摘要：用户打了10次招呼"
    assert short_term.get_input_tokens("s1") == 0  # compact 后重置


def test_compact_preserves_accumulation(short_term: ShortTermMemory) -> None:
    """compact 后可以继续累积消息。"""
    short_term.add("s1", "user", "u0")
    short_term.add("s1", "assistant", "a0")
    short_term.compact("s1", split=1, new_summary="摘要")

    short_term.add("s1", "user", "u1")
    msgs = short_term.get("s1")
    assert len(msgs) == 2  # a0 + u1
    assert msgs[0]["content"] == "a0"
    assert short_term.get_summary("s1") == "摘要"


def test_messages_accumulate_without_limit(short_term: ShortTermMemory) -> None:
    """消息不再有硬上限，可以一直累积。"""
    for i in range(100):
        short_term.add("s1", "user", f"msg{i}")
    msgs = short_term.get("s1")
    assert len(msgs) == 100
    assert msgs[0]["content"] == "msg0"
    assert msgs[99]["content"] == "msg99"


def test_max_sessions_eviction() -> None:
    """超过 500 个 session 时，移除最旧的。"""
    mem = ShortTermMemory()
    for i in range(501):
        mem.add(f"s{i}", "user", f"msg{i}")
    # s0 被移除
    assert mem.get("s0") == []
    assert len(mem.get("s1")) == 1
    assert len(mem.get("s500")) == 1
```

- [ ] **Step 7: 运行测试确认全部通过**

Run: `uv run pytest tests/test_short_term.py -v`
Expected: 11 PASSED

- [ ] **Step 8: 提交**

```bash
git add src/memory/short_term.py tests/test_short_term.py tests/conftest.py
git commit -m "refactor: rewrite ShortTermMemory with list accumulation and compact support"
```

---

### Task 2: 更新 config.py

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: 修改 config**

在 `src/config.py` 中，替换 `short_term_max_rounds` 为新字段：

```python
class BotConfig(BaseModel):
    # LLM
    llm_base_url: str = "http://127.0.0.1:34567/v1"
    llm_api_key: str = "sk-placeholder"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_context_tokens: int = 200_000
    compact_ratio: float = 0.7

    # Memory
    memory_dir: str = "data/memories"

    # Identity
    identities_file: str = "identities.md"

    # NapCat HTTP API
    napcat_api_url: str = "http://localhost:29300"

    # Superusers (for admin tool authorization)
    superusers: set[str] = set()
```

- [ ] **Step 2: 运行全量测试确认无破坏**

Run: `uv run pytest -v`
Expected: 全部 PASSED（其他测试文件不直接引用 `short_term_max_rounds`）

- [ ] **Step 3: 提交**

```bash
git add src/config.py
git commit -m "feat: add llm_max_context_tokens and compact_ratio config, remove short_term_max_rounds"
```

---

### Task 3: _call_api 返回 input_tokens

**Files:**
- Modify: `src/llm/client.py:46-123`

- [ ] **Step 1: 修改 _call_api 返回值**

在 `src/llm/client.py` 的 `_call_api` 函数末尾，将 return 改为：

```python
    return {"text": "".join(text_parts), "tool_uses": tool_uses, "input_tokens": input_tokens + cache_read + cache_create}
```

注意用 `input_tokens + cache_read + cache_create` 作为总量，因为 `input_tokens` 字段只是未命中 cache 的部分，我们需要完整的输入 token 数来判断是否需要 compact。

- [ ] **Step 2: 运行现有测试确认无破坏**

Run: `uv run pytest -v`
Expected: 全部 PASSED

- [ ] **Step 3: 提交**

```bash
git add src/llm/client.py
git commit -m "feat: include input_tokens in _call_api return value"
```

---

### Task 4: 改造 LLMClient.chat() 消息拼装 + compact

**Files:**
- Modify: `src/llm/client.py:126-230`

- [ ] **Step 1: 更新 LLMClient 构造函数**

```python
class LLMClient:
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
    ) -> None:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120, sock_read=30))
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._prompt = prompt_builder
        self._short_term = short_term
        self._tools = tools
        self._max_context_tokens = max_context_tokens
        self._compact_ratio = compact_ratio
```

- [ ] **Step 2: 重写 chat() 方法的消息拼装逻辑**

替换 `chat()` 方法中从 `messages: list[Any] = []` 到工具循环之前的部分：

```python
    async def chat(
        self,
        session_id: str,
        user_id: str,
        user_text: str,
        identity: Identity,
        group_id: str | None = None,
        ctx: ToolContext | None = None,
    ) -> str:
        logger.info("chat | session={} user={} identity={} text={!r}", session_id, user_id, identity.id, user_text[:80])
        self._short_term.add(session_id, "user", user_text)

        # compact 检查
        if self._short_term.needs_compact(session_id, self._max_context_tokens, self._compact_ratio):
            await self._compact(session_id)

        system_blocks = await self._prompt.build_blocks(identity=identity, user_id=user_id, group_id=group_id)

        messages: list[Any] = []

        # 摘要 → 稳定前缀
        summary = self._short_term.get_summary(session_id)
        if summary:
            messages.append({
                "role": "user",
                "content": [_cached_text(f"[对话摘要]\n{summary}")],
            })
            messages.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

            # 群聊上下文（仅在有摘要时注入 messages）
            if group_id:
                ctx_text = self._prompt.build_context_message(group_id)
                if ctx_text:
                    messages.append({
                        "role": "user",
                        "content": [_cached_text(f"[群聊上下文]\n{ctx_text}")],
                    })
                    messages.append({"role": "assistant", "content": "好的，我已了解最近的群聊内容。"})

        # 对话历史（完整，不截断）
        history = self._short_term.get(session_id)
        for i, msg in enumerate(history):
            m = _to_anthropic_message(msg)
            if i == len(history) - 2:
                m = {"role": m["role"], "content": [_cached_text(m["content"])]}
            messages.append(m)

        # ... 工具循环保持不变 ...
```

- [ ] **Step 3: 在工具循环末尾记录 token 数**

在 `chat()` 方法中，工具循环内 `if not tool_uses:` 分支和方法末尾（`tool loop exhausted`）都需要记录 token 数。

修改工具循环内的 return 分支：

```python
            if not tool_uses:
                reply = text or "..."
                logger.info("reply | session={} len={}", session_id, len(reply))
                self._short_term.add(session_id, "assistant", reply)
                self._short_term.set_input_tokens(session_id, result["input_tokens"])
                return reply
```

修改方法末尾（工具循环耗尽后）：

```python
        logger.warning("tool loop exhausted | session={} rounds={}", session_id, MAX_TOOL_ROUNDS)
        result = await self._call(system_blocks, messages)
        reply = result["text"] or "..."
        self._short_term.add(session_id, "assistant", reply)
        self._short_term.set_input_tokens(session_id, result["input_tokens"])
        return reply
```

- [ ] **Step 4: 添加 _compact() 方法**

在 `LLMClient` 类中，`chat()` 方法之后添加：

```python
    async def _compact(self, session_id: str) -> None:
        """将历史前半部分压缩成摘要。"""
        history = self._short_term.get(session_id)
        if len(history) < 4:
            return  # 消息太少，不值得 compact

        old_summary = self._short_term.get_summary(session_id)
        split = len(history) // 2

        # 拼装要压缩的内容
        lines: list[str] = []
        if old_summary:
            lines.append(f"[之前的对话摘要]\n{old_summary}\n")
        for msg in history[:split]:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}: {msg['content']}")
        conversation_text = "\n".join(lines)

        system = [{"type": "text", "text": (
            "你是一个对话压缩助手。请将以下对话历史压缩成简洁的中文摘要。"
            "保留关键信息：用户的问题、重要决策、关键结论、用户偏好。"
            "去掉寒暄、重复内容和过程性细节。输出纯摘要文本，不要加标题或格式。"
        )}]
        compress_messages = [{"role": "user", "content": conversation_text}]

        logger.info("compact | session={} split={}/{}", session_id, split, len(history))
        result = await _call_api(
            self._session, self._base_url, self._api_key, self._model,
            system, compress_messages, max_tokens=1024,
        )
        new_summary = result["text"].strip()
        if new_summary:
            self._short_term.compact(session_id, split, new_summary)
            logger.info("compact done | session={} summary_len={}", session_id, len(new_summary))
        else:
            logger.warning("compact produced empty summary | session={}", session_id)
```

- [ ] **Step 5: 运行全量测试确认无破坏**

Run: `uv run pytest -v`
Expected: 全部 PASSED

- [ ] **Step 6: 提交**

```bash
git add src/llm/client.py
git commit -m "feat: add compact support to LLMClient with token-based triggering"
```

---

### Task 5: 更新插件初始化

**Files:**
- Modify: `src/plugins/chat/__init__.py:36-66`

- [ ] **Step 1: 修改 _init() 函数**

将 `ShortTermMemory` 和 `LLMClient` 的初始化更新为：

```python
@driver.on_startup
async def _init() -> None:
    global _llm, _identity_mgr, _group_ctx, _short_term

    long_term = LongTermMemory(memory_dir=bot_config.memory_dir)
    _short_term = ShortTermMemory()
    _group_ctx = GroupContext()
    prompt_builder = PromptBuilder(long_term=long_term, group_context=_group_ctx)
    short_term = _short_term

    superusers = bot_config.superusers | driver.config.superusers

    tools = ToolRegistry()
    tools.register(SaveMemoryTool(long_term))
    tools.register(RecallMemoryTool(long_term))
    tools.register(DateTimeTool())
    tools.register(WebFetchTool())
    tools.register(HttpApiTool())
    tools.register(MuteUserTool(superusers))
    tools.register(SetTitleTool(superusers))
    tools.register(SendGroupMsgTool(superusers))

    _identity_mgr = IdentityManager()
    await _identity_mgr.load_file(bot_config.identities_file)

    _llm = LLMClient(
        base_url=bot_config.llm_base_url,
        api_key=bot_config.llm_api_key,
        model=bot_config.llm_model,
        prompt_builder=prompt_builder,
        short_term=short_term,
        tools=tools,
        max_context_tokens=bot_config.llm_max_context_tokens,
        compact_ratio=bot_config.compact_ratio,
    )
```

- [ ] **Step 2: 运行全量测试**

Run: `uv run pytest -v`
Expected: 全部 PASSED

- [ ] **Step 3: 运行 lint 和类型检查**

Run: `uv run ruff check src/ && uv run pyright`
Expected: 无错误

- [ ] **Step 4: 提交**

```bash
git add src/plugins/chat/__init__.py
git commit -m "feat: wire up compact config in plugin initialization"
```

---

### Task 6: 最终验证

- [ ] **Step 1: 运行全量测试套件**

Run: `uv run pytest -v`
Expected: 全部 PASSED

- [ ] **Step 2: 运行 lint**

Run: `uv run ruff check src/`
Expected: 无错误

- [ ] **Step 3: 运行类型检查**

Run: `uv run pyright`
Expected: 无错误

- [ ] **Step 4: 检查 git log 确认提交链完整**

Run: `git log --oneline -6`
Expected: 看到 Task 1-5 的 4 个提交
