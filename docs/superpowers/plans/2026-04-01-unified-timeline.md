# 群聊统一时间线 + 缓存预热 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 合并 GroupContext + ShortTermMemory 为统一时间线，解决 Anthropic 缓存前缀失效问题，新增缓存预热和 TOML 配置体系。

**Architecture:** 新增 `GroupTimeline` 统一管理群聊消息流（所有群员 + bot），连续非 bot 消息合并为 user 块发给 API。`ShortTermMemory` 仅保留给私聊。配置从 `.env` 迁移到 TOML + CLI + 环境变量三层覆盖。预热通过 `max_tokens=1` 的异步 API 调用保持缓存热度。

**Tech Stack:** Python 3.12, Pydantic v2, tomllib (stdlib), argparse, aiohttp, NoneBot2, Anthropic Messages API

---

### Task 1: TOML 配置体系 — BotConfig + config_loader

**Files:**
- Modify: `src/config.py`
- Create: `src/config_loader.py`
- Create: `config.example.toml`
- Test: `tests/test_config_loader.py`

- [ ] **Step 1: 写 config_loader 的测试**

```python
# tests/test_config_loader.py
import os
from pathlib import Path

from src.config import BotConfig
from src.config_loader import load_config


def test_load_defaults_without_file() -> None:
    """无 TOML 文件时使用默认值。"""
    cfg = load_config(config_path=None, cli_overrides={})
    assert cfg.llm.model == "claude-sonnet-4-20250514"
    assert cfg.llm.cache.warm_enabled is True
    assert cfg.llm.cache.warm_interval_messages == 10
    assert cfg.group.max_timeline_messages == 200


def test_load_from_toml(tmp_path: Path) -> None:
    """从 TOML 文件加载。"""
    toml_file = tmp_path / "config.toml"
    toml_file.write_text("""
[llm]
model = "claude-haiku-3"

[llm.cache]
warm_interval_messages = 5

[group]
max_timeline_messages = 100
""")
    cfg = load_config(config_path=str(toml_file), cli_overrides={})
    assert cfg.llm.model == "claude-haiku-3"
    assert cfg.llm.cache.warm_interval_messages == 5
    assert cfg.group.max_timeline_messages == 100
    # 未指定的保持默认
    assert cfg.llm.cache.warm_ttl_seconds == 300


def test_env_overrides_toml(tmp_path: Path, monkeypatch: object) -> None:
    """环境变量覆盖 TOML 值。"""
    toml_file = tmp_path / "config.toml"
    toml_file.write_text('[llm]\nmodel = "from-toml"')
    monkeypatch.setenv("LLM_MODEL", "from-env")  # type: ignore[attr-defined]
    cfg = load_config(config_path=str(toml_file), cli_overrides={})
    assert cfg.llm.model == "from-env"


def test_cli_overrides_env(monkeypatch: object) -> None:
    """CLI 参数覆盖环境变量。"""
    monkeypatch.setenv("LLM_MODEL", "from-env")  # type: ignore[attr-defined]
    cfg = load_config(config_path=None, cli_overrides={"llm_model": "from-cli"})
    assert cfg.llm.model == "from-cli"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config_loader'`

- [ ] **Step 3: 重构 BotConfig 为嵌套 Pydantic 模型**

```python
# src/config.py
"""Bot 配置模型：嵌套 Pydantic v2，从 TOML / 环境变量 / CLI 加载。"""

from pydantic import BaseModel


class CacheConfig(BaseModel):
    warm_enabled: bool = True
    warm_interval_messages: int = 10
    warm_ttl_seconds: int = 300


class ContextConfig(BaseModel):
    max_context_tokens: int = 200_000
    compact_ratio: float = 0.7


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:34567/v1"
    api_key: str = "sk-placeholder"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    context: ContextConfig = ContextConfig()
    cache: CacheConfig = CacheConfig()


class MemoryConfig(BaseModel):
    dir: str = "data/memories"


class IdentityConfig(BaseModel):
    file: str = "identities.md"


class GroupConfig(BaseModel):
    max_timeline_messages: int = 200
    history_load_count: int = 30


class NapcatConfig(BaseModel):
    api_url: str = "http://localhost:29300"


class BotConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    memory: MemoryConfig = MemoryConfig()
    identity: IdentityConfig = IdentityConfig()
    group: GroupConfig = GroupConfig()
    napcat: NapcatConfig = NapcatConfig()
    superusers: set[str] = set()
```

- [ ] **Step 4: 实现 config_loader**

```python
# src/config_loader.py
"""TOML + 环境变量 + CLI 三层配置加载。

优先级: CLI > 环境变量 > TOML > 默认值
"""

import os
import tomllib
from pathlib import Path
from typing import Any

from src.config import BotConfig

# 环境变量名 → BotConfig 字段路径
_ENV_MAP: dict[str, str] = {
    "LLM_BASE_URL": "llm.base_url",
    "LLM_API_KEY": "llm.api_key",
    "LLM_MODEL": "llm.model",
    "NAPCAT_API_URL": "napcat.api_url",
}

# CLI 参数名 → BotConfig 字段路径
_CLI_MAP: dict[str, str] = {
    "llm_base_url": "llm.base_url",
    "llm_api_key": "llm.api_key",
    "llm_model": "llm.model",
    "config": "",  # 特殊处理，不映射到 BotConfig
}


def _deep_set(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """在嵌套 dict 中按 'a.b.c' 路径设值。"""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(
    config_path: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> BotConfig:
    """加载配置，按 TOML → 环境变量 → CLI 逐层覆盖。"""
    data: dict[str, Any] = {}

    # 1. TOML 文件
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)

    # 2. 环境变量覆盖
    for env_key, field_path in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is not None:
            _deep_set(data, field_path, val)

    # 3. CLI 覆盖
    if cli_overrides:
        for cli_key, val in cli_overrides.items():
            if val is None:
                continue
            field_path = _CLI_MAP.get(cli_key)
            if field_path:
                _deep_set(data, field_path, val)

    return BotConfig(**data)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: 4 passed

- [ ] **Step 6: 创建 config.example.toml**

```toml
# config.example.toml
# QQ Bot 配置文件
# 复制为 config.toml 后按需修改
# 优先级: CLI 参数 > 环境变量 > 此文件 > 默认值

[llm]
base_url = "http://127.0.0.1:34567/v1"     # 环境变量: LLM_BASE_URL | CLI: --llm-base-url
api_key = "sk-placeholder"                   # 环境变量: LLM_API_KEY  | CLI: --llm-api-key  (建议用环境变量)
model = "claude-sonnet-4-20250514"           # 环境变量: LLM_MODEL    | CLI: --llm-model
max_tokens = 1024

[llm.context]
max_context_tokens = 200_000                 # 上下文窗口上限
compact_ratio = 0.7                          # input_tokens > max * ratio 时触发 compact

[llm.cache]
warm_enabled = true                          # 是否启用缓存预热
warm_interval_messages = 10                  # 每积累 N 条新群消息触发一次预热
warm_ttl_seconds = 300                       # 距上次 API 调用超过此秒数不预热（缓存已过期）

[memory]
dir = "data/memories"

[identity]
file = "identities.md"

[group]
max_timeline_messages = 200                  # 群时间线最大消息条数（含 bot 回复）
history_load_count = 30                      # 启动时从 NapCat 拉取的历史消息数

[napcat]
api_url = "http://localhost:29300"           # 环境变量: NAPCAT_API_URL

[bot]
superusers = []
```

- [ ] **Step 7: Commit**

```bash
git add src/config.py src/config_loader.py config.example.toml tests/test_config_loader.py
git commit -m "feat: TOML config system with env/CLI overrides"
```

---

### Task 2: GroupTimeline — 数据模型和消息转换

**Files:**
- Create: `src/memory/group_timeline.py`
- Test: `tests/test_group_timeline.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: 写 GroupTimeline 基础测试**

```python
# tests/test_group_timeline.py
from src.memory.group_timeline import GroupTimeline


def test_add_and_get_messages(group_timeline: GroupTimeline) -> None:
    group_timeline.add("g1", role="user", speaker="小明(100)", content="你好")
    group_timeline.add("g1", role="user", speaker="小红(200)", content="你好呀")
    msgs = group_timeline.get_messages("g1")
    assert len(msgs) == 2
    assert msgs[0]["speaker"] == "小明(100)"
    assert msgs[1]["content"] == "你好呀"


def test_to_anthropic_merges_consecutive_users(group_timeline: GroupTimeline) -> None:
    """连续 user 消息合并为一个 Anthropic user 块。"""
    group_timeline.add("g1", role="user", speaker="小明(100)", content="天气好")
    group_timeline.add("g1", role="user", speaker="小红(200)", content="是啊")
    group_timeline.add("g1", role="assistant", content="确实不错")
    group_timeline.add("g1", role="user", speaker="小明(100)", content="明天呢")

    result = group_timeline.to_anthropic_messages("g1")
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert "小明(100): 天气好" in result[0]["content"]
    assert "小红(200): 是啊" in result[0]["content"]
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "确实不错"
    assert result[2]["role"] == "user"
    assert "小明(100): 明天呢" in result[2]["content"]


def test_to_anthropic_empty(group_timeline: GroupTimeline) -> None:
    assert group_timeline.to_anthropic_messages("nonexistent") == []


def test_group_isolation(group_timeline: GroupTimeline) -> None:
    group_timeline.add("g1", role="user", speaker="A(1)", content="群1")
    group_timeline.add("g2", role="user", speaker="B(2)", content="群2")
    m1 = group_timeline.get_messages("g1")
    m2 = group_timeline.get_messages("g2")
    assert len(m1) == 1
    assert m1[0]["content"] == "群1"
    assert len(m2) == 1
    assert m2[0]["content"] == "群2"


def test_max_messages_eviction() -> None:
    tl = GroupTimeline(max_messages=3)
    for i in range(5):
        tl.add("g1", role="user", speaker=f"u({i})", content=f"msg{i}")
    msgs = tl.get_messages("g1")
    assert len(msgs) == 3
    assert msgs[0]["content"] == "msg2"


def test_assistant_no_speaker(group_timeline: GroupTimeline) -> None:
    group_timeline.add("g1", role="assistant", content="我是bot")
    msgs = group_timeline.get_messages("g1")
    assert msgs[0]["speaker"] is None
    assert msgs[0]["role"] == "assistant"
```

- [ ] **Step 2: 添加 conftest fixture**

在 `tests/conftest.py` 末尾添加:

```python
from src.memory.group_timeline import GroupTimeline


@pytest.fixture
def group_timeline() -> GroupTimeline:
    return GroupTimeline(max_messages=50)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_group_timeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.memory.group_timeline'`

- [ ] **Step 4: 实现 GroupTimeline**

```python
# src/memory/group_timeline.py
"""群聊统一时间线：所有群消息（含 bot 回复）按时间顺序存储。

取代 GroupContext + ShortTermMemory（群聊部分）。
连续 user 消息合并为一个 Anthropic user 块，bot 回复作为 assistant 块。
"""

import time
from collections import defaultdict
from typing import Any, Literal, TypedDict

_MAX_GROUPS = 200


class TimelineMessage(TypedDict):
    role: Literal["user", "assistant"]
    speaker: str | None  # user 时为 "昵称(QQ号)"，assistant 时为 None
    content: str


class _GroupState:
    __slots__ = ("_max", "last_api_call_time", "last_input_tokens", "messages", "new_msg_count", "summary")

    def __init__(self, max_messages: int) -> None:
        self._max = max_messages
        self.messages: list[TimelineMessage] = []
        self.summary: str = ""
        self.last_input_tokens: int = 0
        self.new_msg_count: int = 0
        self.last_api_call_time: float = 0.0


class GroupTimeline:
    def __init__(self, max_messages: int = 200) -> None:
        self._max = max_messages
        self._store: dict[str, _GroupState] = {}

    def _get_or_create(self, group_id: str) -> _GroupState:
        if group_id not in self._store:
            if len(self._store) >= _MAX_GROUPS:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[group_id] = _GroupState(self._max)
        return self._store[group_id]

    def add(
        self,
        group_id: str,
        *,
        role: Literal["user", "assistant"],
        content: str,
        speaker: str | None = None,
    ) -> None:
        state = self._get_or_create(group_id)
        state.messages.append(TimelineMessage(role=role, speaker=speaker, content=content))
        # 溢出时丢弃最早的
        while len(state.messages) > self._max:
            state.messages.pop(0)
        if role == "user":
            state.new_msg_count += 1

    def get_messages(self, group_id: str) -> list[TimelineMessage]:
        if group_id not in self._store:
            return []
        return list(self._store[group_id].messages)

    def to_anthropic_messages(self, group_id: str) -> list[dict[str, str]]:
        """将时间线转为 Anthropic messages 格式，连续 user 合并。"""
        messages = self.get_messages(group_id)
        if not messages:
            return []

        result: list[dict[str, str]] = []
        current_user_parts: list[str] = []

        for msg in messages:
            if msg["role"] == "user":
                line = f"{msg['speaker']}: {msg['content']}" if msg["speaker"] else msg["content"]
                current_user_parts.append(line)
            else:
                # 遇到 assistant，先 flush 积累的 user 消息
                if current_user_parts:
                    result.append({"role": "user", "content": "\n".join(current_user_parts)})
                    current_user_parts = []
                result.append({"role": "assistant", "content": msg["content"]})

        # flush 尾部 user 消息
        if current_user_parts:
            result.append({"role": "user", "content": "\n".join(current_user_parts)})

        return result

    # ── compact 相关状态 ──

    def get_summary(self, group_id: str) -> str:
        if group_id not in self._store:
            return ""
        return self._store[group_id].summary

    def set_input_tokens(self, group_id: str, tokens: int) -> None:
        if group_id in self._store:
            state = self._store[group_id]
            state.last_input_tokens = tokens
            state.last_api_call_time = time.monotonic()
            state.new_msg_count = 0

    def get_input_tokens(self, group_id: str) -> int:
        if group_id not in self._store:
            return 0
        return self._store[group_id].last_input_tokens

    def needs_compact(self, group_id: str, max_tokens: int, ratio: float) -> bool:
        return self.get_input_tokens(group_id) > max_tokens * ratio

    def compact(self, group_id: str, split: int, new_summary: str) -> None:
        if group_id not in self._store:
            return
        state = self._store[group_id]
        state.messages = state.messages[split:]
        state.summary = new_summary
        state.last_input_tokens = 0

    # ── 预热状态 ──

    def should_warm(self, group_id: str, interval: int, ttl: float) -> bool:
        """是否应该预热缓存。"""
        if group_id not in self._store:
            return False
        state = self._store[group_id]
        if state.new_msg_count < interval:
            return False
        if state.last_api_call_time == 0.0:
            return False  # 从未调过 API，没有缓存可续
        return (time.monotonic() - state.last_api_call_time) < ttl

    def reset_warm_counter(self, group_id: str) -> None:
        if group_id in self._store:
            self._store[group_id].new_msg_count = 0
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_group_timeline.py -v`
Expected: 6 passed

- [ ] **Step 6: 追加预热状态测试**

在 `tests/test_group_timeline.py` 末尾添加:

```python
def test_should_warm_basic(group_timeline: GroupTimeline) -> None:
    """积累足够消息且在 TTL 内 → 应该预热。"""
    # 模拟一次 API 调用
    group_timeline.add("g1", role="user", speaker="A(1)", content="hi")
    group_timeline.set_input_tokens("g1", 1000)

    # 积累 10 条新消息
    for i in range(10):
        group_timeline.add("g1", role="user", speaker="A(1)", content=f"msg{i}")

    assert group_timeline.should_warm("g1", interval=10, ttl=300)


def test_should_warm_not_enough_messages(group_timeline: GroupTimeline) -> None:
    group_timeline.add("g1", role="user", speaker="A(1)", content="hi")
    group_timeline.set_input_tokens("g1", 1000)

    for i in range(5):
        group_timeline.add("g1", role="user", speaker="A(1)", content=f"msg{i}")

    assert not group_timeline.should_warm("g1", interval=10, ttl=300)


def test_should_warm_never_called_api(group_timeline: GroupTimeline) -> None:
    """从未调过 API → 没有缓存可续，不预热。"""
    for i in range(20):
        group_timeline.add("g1", role="user", speaker="A(1)", content=f"msg{i}")

    assert not group_timeline.should_warm("g1", interval=10, ttl=300)


def test_reset_warm_counter(group_timeline: GroupTimeline) -> None:
    group_timeline.add("g1", role="user", speaker="A(1)", content="hi")
    group_timeline.set_input_tokens("g1", 1000)

    for i in range(10):
        group_timeline.add("g1", role="user", speaker="A(1)", content=f"msg{i}")

    group_timeline.reset_warm_counter("g1")
    assert not group_timeline.should_warm("g1", interval=10, ttl=300)


def test_compact(group_timeline: GroupTimeline) -> None:
    for i in range(10):
        group_timeline.add("g1", role="user", speaker=f"u({i})", content=f"msg{i}")
    group_timeline.set_input_tokens("g1", 99999)

    group_timeline.compact("g1", split=5, new_summary="前5条摘要")
    msgs = group_timeline.get_messages("g1")
    assert len(msgs) == 5
    assert msgs[0]["content"] == "msg5"
    assert group_timeline.get_summary("g1") == "前5条摘要"
    assert group_timeline.get_input_tokens("g1") == 0
```

- [ ] **Step 7: 运行全部 timeline 测试**

Run: `uv run pytest tests/test_group_timeline.py -v`
Expected: 11 passed

- [ ] **Step 8: Commit**

```bash
git add src/memory/group_timeline.py tests/test_group_timeline.py tests/conftest.py
git commit -m "feat: GroupTimeline with message merging and warm state"
```

---

### Task 3: LLMClient 改造 — 群聊走 GroupTimeline

**Files:**
- Modify: `src/llm/client.py`
- Modify: `src/llm/prompt.py`

- [ ] **Step 1: 修改 LLMClient.__init__ 接受 GroupTimeline**

在 `src/llm/client.py` 中，修改 `LLMClient.__init__` 签名，增加 `group_timeline` 参数和 cache 配置参数：

```python
# src/llm/client.py — 修改导入和 __init__

# 新增导入
from src.memory.group_timeline import GroupTimeline

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
        group_timeline: GroupTimeline | None = None,
        long_term: "LongTermMemory | None" = None,
        warm_enabled: bool = True,
        warm_interval_messages: int = 10,
        warm_ttl_seconds: int = 300,
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
        self._timeline = group_timeline
        self._long_term = long_term
        self._warm_enabled = warm_enabled
        self._warm_interval = warm_interval_messages
        self._warm_ttl = warm_ttl_seconds
        self._warming = False  # 防并发 flag
```

- [ ] **Step 2: 重写 chat() 方法，群聊走 GroupTimeline**

替换 `src/llm/client.py` 中的 `chat()` 方法：

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

        is_group = group_id is not None and self._timeline is not None

        if is_group:
            # 群聊：消息已由 group_listener 写入 timeline，此处不再重复 add
            tl = self._timeline
            if tl.needs_compact(group_id, self._max_context_tokens, self._compact_ratio):
                await self._compact_group(group_id, identity)
            system_blocks = await self._prompt.build_blocks(identity=identity, user_id=user_id, group_id=group_id)
            messages = self._build_group_messages(group_id)
        else:
            # 私聊：沿用 ShortTermMemory
            self._short_term.add(session_id, "user", user_text)
            if self._short_term.needs_compact(session_id, self._max_context_tokens, self._compact_ratio):
                await self._compact(session_id)
            system_blocks = await self._prompt.build_blocks(identity=identity, user_id=user_id, group_id=group_id)
            messages = self._build_private_messages(session_id, group_id)

        tool_defs: list[dict[str, Any]] | None = None
        if not self._tools.empty:
            tool_defs = _to_anthropic_tools(self._tools.to_openai_tools())

        for round_i in range(MAX_TOOL_ROUNDS):
            result = await self._call(system_blocks, messages, tools=tool_defs)
            text: str = result["text"]
            tool_uses: list[_ToolUse] = result["tool_uses"]

            if not tool_uses:
                reply = text or "..."
                logger.info("reply | session={} len={}", session_id, len(reply))
                if is_group:
                    tl.add(group_id, role="assistant", content=reply)
                    tl.set_input_tokens(group_id, result["input_tokens"])
                else:
                    self._short_term.add(session_id, "assistant", reply)
                    self._short_term.set_input_tokens(session_id, result["input_tokens"])
                return reply

            for tu in tool_uses:
                logger.info(
                    "tool_call | round={} name={} args={!r}",
                    round_i, tu.name, json.dumps(tu.input, ensure_ascii=False)[:200],
                )

            assistant_content: list[dict[str, Any]] = []
            if text:
                assistant_content.append({"type": "text", "text": text})
            for tu in tool_uses:
                assistant_content.append({"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input})
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                tool_ctx = ctx or ToolContext(user_id=user_id, group_id=group_id)
                tool_result = await self._tools.call(tu.name, json.dumps(tu.input), ctx=tool_ctx)
                logger.debug("tool_result | name={} result={!r}", tu.name, tool_result[:200])
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": tool_result})
            messages.append({"role": "user", "content": tool_results})

        logger.warning("tool loop exhausted | session={} rounds={}", session_id, MAX_TOOL_ROUNDS)
        result = await self._call(system_blocks, messages)
        reply = result["text"] or "..."
        if is_group:
            tl.add(group_id, role="assistant", content=reply)
            tl.set_input_tokens(group_id, result["input_tokens"])
        else:
            self._short_term.add(session_id, "assistant", reply)
            self._short_term.set_input_tokens(session_id, result["input_tokens"])
        return reply
```

- [ ] **Step 3: 添加 _build_group_messages 和 _build_private_messages 方法**

在 `LLMClient` 类中添加：

```python
    def _build_group_messages(self, group_id: str) -> list[Any]:
        """群聊：从 GroupTimeline 构建 messages。"""
        messages: list[Any] = []

        summary = self._timeline.get_summary(group_id)
        if summary:
            messages.append({
                "role": "user",
                "content": [_cached_text(f"[对话摘要]\n{summary}")],
            })
            messages.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

        anthropic_msgs = self._timeline.to_anthropic_messages(group_id)
        for i, m in enumerate(anthropic_msgs):
            if i == len(anthropic_msgs) - 2:
                # 倒数第二条加 cache_control
                m = {"role": m["role"], "content": [_cached_text(m["content"])]}
            messages.append(m)

        return messages

    def _build_private_messages(self, session_id: str, group_id: str | None = None) -> list[Any]:
        """私聊：沿用 ShortTermMemory，保持原有逻辑。"""
        messages: list[Any] = []

        summary = self._short_term.get_summary(session_id)
        if summary:
            messages.append({
                "role": "user",
                "content": [_cached_text(f"[对话摘要]\n{summary}")],
            })
            messages.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

        history = self._short_term.get(session_id)
        for i, msg in enumerate(history):
            m = _to_anthropic_message(msg)
            if i == len(history) - 2:
                m = {"role": m["role"], "content": [_cached_text(m["content"])]}
            messages.append(m)

        return messages
```

- [ ] **Step 4: 删除 prompt.py 中的 build_context_message**

在 `src/llm/prompt.py` 中删除 `build_context_message` 方法和 `GroupContext` 导入。修改后的文件：

```python
# src/llm/prompt.py
"""Soul 层：动态拼装 System Prompt，分层支持 context caching。

缓存策略：
  system blocks（带 cache_control，跨请求复用）：
    1. 人设性格 + 工具指南  → 几乎不变
    2. 用户记忆(.qmd)       → 偶尔更新
"""

from typing import Any

from src.identity.models import Identity
from src.memory.long_term import LongTermMemory

TOOL_GUIDE = """\
【工具使用指南】
- 当用户提到自己的信息（昵称、爱好、经历等）时，用 save_memory 记住
- 当需要回忆用户信息时，用 recall_memory 查询
- 当用户问时间日期时，用 get_datetime 获取
- 当需要查网页内容时，用 web_fetch 抓取
- 当需要调外部 API 时，用 http_api 调用
- 群管理操作（禁言、头衔、发消息）用对应的群管理工具
"""


class PromptBuilder:
    def __init__(self, long_term: LongTermMemory) -> None:
        self._long_term = long_term

    async def build_blocks(self, identity: Identity, user_id: str, group_id: str | None = None) -> list[dict[str, Any]]:
        """返回 system blocks，只含稳定内容，最大化 cache 命中。"""
        blocks: list[dict[str, Any]] = []

        base_text = identity.personality + "\n\n" + TOOL_GUIDE
        if group_id:
            base_text += f"\n【当前在群 {group_id} 中对话】"
        blocks.append({"type": "text", "text": base_text, "cache_control": {"type": "ephemeral"}})

        memory_ctx = await self._long_term.get_full_context(user_id)
        if memory_ctx.strip():
            blocks.append({
                "type": "text",
                "text": f"【关于当前用户 {user_id} 的记忆】\n{memory_ctx.strip()}",
                "cache_control": {"type": "ephemeral"},
            })

        return blocks
```

- [ ] **Step 5: 运行 lint 和现有测试**

Run: `uv run ruff check src/ && uv run pytest tests/test_prompt.py tests/test_group_timeline.py -v`
Expected: lint clean, tests pass

- [ ] **Step 6: Commit**

```bash
git add src/llm/client.py src/llm/prompt.py
git commit -m "feat: LLMClient uses GroupTimeline for group chats"
```

---

### Task 4: Compact 增强 — 提取长期记忆

> **Note:** Task 3 的 `chat()` 调用了 `self._compact_group()`，在本 Task 实现。两个 Task 应连续完成。

**Files:**
- Modify: `src/llm/client.py`

- [ ] **Step 1: 添加 _compact_group 方法**

在 `LLMClient` 类中，添加群聊 compact 方法（保留原 `_compact` 给私聊）：

```python
    async def _compact_group(self, group_id: str, identity: Identity) -> None:
        """群聊 compact：压缩时间线前半，同时提取长期记忆。"""
        tl = self._timeline
        messages = tl.get_messages(group_id)
        if len(messages) < 4:
            return

        old_summary = tl.get_summary(group_id)
        split = len(messages) // 2

        lines: list[str] = []
        if old_summary:
            lines.append(f"[之前的对话摘要]\n{old_summary}\n")
        for msg in messages[:split]:
            speaker = msg["speaker"] or identity.name
            lines.append(f"{speaker}: {msg['content']}")
        conversation_text = "\n".join(lines)

        system = [{"type": "text", "text": (
            "你是一个对话分析助手。请完成两个任务：\n"
            "1. 将以下群聊记录压缩成简洁的中文摘要。保留关键信息：讨论话题、重要决策、关键结论。\n"
            "2. 提取值得长期记住的用户信息（昵称、爱好、特征、关键事件）。\n\n"
            '以 JSON 格式输出：\n'
            '{"summary": "摘要文本", "memories": [{"user_id": "QQ号", "nickname": "昵称", '
            '"traits": {"key": "value"}, "event": "事件描述"}]}\n\n'
            "如果没有值得记住的用户信息，memories 为空数组。只输出 JSON，不要加其他文字。"
        )}]
        compress_messages = [{"role": "user", "content": conversation_text}]

        logger.info("compact | group={} split={}/{}", group_id, split, len(messages))
        result = await _call_api(
            self._session, self._base_url, self._api_key, self._model,
            system, compress_messages, max_tokens=1024,
        )
        raw = result["text"].strip()

        # 解析 JSON 输出
        try:
            parsed = json.loads(raw)
            new_summary = parsed.get("summary", "")
            memories = parsed.get("memories", [])
        except json.JSONDecodeError:
            # fallback: 当作纯摘要文本
            new_summary = raw
            memories = []

        if new_summary:
            tl.compact(group_id, split, new_summary)
            logger.info("compact done | group={} summary_len={}", group_id, len(new_summary))
        else:
            logger.warning("compact produced empty summary | group={}", group_id)

        # 写入长期记忆
        if memories and self._long_term:
            for mem in memories:
                uid = mem.get("user_id", "")
                if not uid:
                    continue
                nickname = mem.get("nickname")
                traits = mem.get("traits")
                event = mem.get("event")
                if nickname or traits:
                    await self._long_term.update_profile(uid, nickname=nickname, traits=traits)
                if event:
                    await self._long_term.add_event(uid, event)
            logger.info("compact memories | group={} count={}", group_id, len(memories))
```

- [ ] **Step 2: 运行 lint**

Run: `uv run ruff check src/llm/client.py`
Expected: clean

- [ ] **Step 3: Commit**

```bash
git add src/llm/client.py
git commit -m "feat: compact extracts long-term memories during summarization"
```

---

### Task 5: 缓存预热

**Files:**
- Modify: `src/llm/client.py`

- [ ] **Step 1: 添加 warm_cache 和 maybe_warm 方法**

在 `LLMClient` 类中添加：

```python
    async def _warm_cache(self, group_id: str, identity: Identity, user_id: str) -> None:
        """max_tokens=1 预热调用，仅建缓存不生成内容。"""
        try:
            system_blocks = await self._prompt.build_blocks(identity=identity, user_id=user_id, group_id=group_id)
            messages = self._build_group_messages(group_id)
            if not messages:
                return

            tool_defs: list[dict[str, Any]] | None = None
            if not self._tools.empty:
                tool_defs = _to_anthropic_tools(self._tools.to_openai_tools())

            await _call_api(
                self._session, self._base_url, self._api_key, self._model,
                system_blocks, messages, max_tokens=1, tools=tool_defs,
            )
            self._timeline.reset_warm_counter(group_id)
            logger.debug("cache warm | group={}", group_id)
        except Exception:
            logger.warning("cache warm failed | group={}", group_id)
        finally:
            self._warming = False

    def maybe_warm(
        self, group_id: str, identity: Identity, user_id: str,
    ) -> bool:
        """检查是否需要预热，如果需要则启动异步任务。返回是否已启动。"""
        if not self._warm_enabled or not self._timeline:
            return False
        if self._warming:
            return False
        if not self._timeline.should_warm(group_id, self._warm_interval, self._warm_ttl):
            return False

        import asyncio
        self._warming = True
        asyncio.create_task(self._warm_cache(group_id, identity, user_id))
        return True
```

- [ ] **Step 2: 运行 lint**

Run: `uv run ruff check src/llm/client.py`
Expected: clean

- [ ] **Step 3: Commit**

```bash
git add src/llm/client.py
git commit -m "feat: async cache warming with max_tokens=1"
```

---

### Task 6: 插件集成 — 接入 GroupTimeline + 新配置

**Files:**
- Modify: `src/plugins/chat/__init__.py`
- Modify: `src/memory/history_loader.py`
- Modify: `bot.py`

- [ ] **Step 1: 改造 history_loader 写入 GroupTimeline**

```python
# src/memory/history_loader.py
"""启动时从 NapCat HTTP API 拉取群历史消息，填充群聊时间线。"""

from typing import Any

import aiohttp
from loguru import logger

from src.memory.group_timeline import GroupTimeline


async def load_group_history(
    napcat_url: str,
    group_ids: list[str],
    timeline: GroupTimeline,
    count: int = 30,
) -> None:
    """从 NapCat 拉取多个群的历史消息。"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        for gid in group_ids:
            try:
                await _load_one_group(session, napcat_url, gid, timeline, count)
            except Exception:
                logger.warning("load_history failed | group={}", gid)


async def _load_one_group(
    session: aiohttp.ClientSession,
    napcat_url: str,
    group_id: str,
    timeline: GroupTimeline,
    count: int,
) -> None:
    async with session.post(
        f"{napcat_url}/get_group_msg_history",
        json={"group_id": int(group_id), "count": count},
    ) as resp:
        data: dict[str, Any] = await resp.json()

    if data.get("retcode") != 0:
        logger.warning("get_group_msg_history error | group={} resp={}", group_id, data.get("message", ""))
        return

    messages: list[dict[str, Any]] = data.get("data", {}).get("messages", [])
    if not messages:
        return

    loaded = 0

    for msg in messages:
        sender: dict[str, Any] = msg.get("sender", {})
        user_id = str(sender.get("user_id", ""))
        nickname = sender.get("nickname", "") or sender.get("card", "") or user_id

        text_parts: list[str] = []
        for seg in msg.get("message", []):
            if seg.get("type") == "text":
                text_parts.append(seg.get("data", {}).get("text", ""))
        text = "".join(text_parts).strip()
        if not text:
            continue

        timeline.add(group_id, role="user", speaker=f"{nickname}({user_id})", content=text)
        loaded += 1

    logger.info("history loaded | group={} messages={}", group_id, loaded)
```

- [ ] **Step 2: 改造 plugin __init__.py**

```python
# src/plugins/chat/__init__.py
"""对话插件：@机器人 触发，Soul + 记忆 + 工具 + 多人设 + 统一时间线。"""

from loguru import logger
from nonebot import get_driver, on_command, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageEvent
from nonebot.params import CommandArg
from nonebot.rule import to_me

from src.config import BotConfig
from src.config_loader import load_config
from src.identity import IdentityManager
from src.llm.client import LLMClient
from src.llm.prompt import PromptBuilder
from src.memory.group_timeline import GroupTimeline
from src.memory.history_loader import load_group_history
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.tools import ToolRegistry
from src.tools.context import ToolContext
from src.tools.datetime_tool import DateTimeTool
from src.tools.group_admin import MuteUserTool, SendGroupMsgTool, SetTitleTool
from src.tools.http_api import HttpApiTool
from src.tools.memory_tool import RecallMemoryTool, SaveMemoryTool
from src.tools.web_fetch import WebFetchTool

driver = get_driver()

_llm: LLMClient
_identity_mgr: IdentityManager
_timeline: GroupTimeline
_short_term: ShortTermMemory


@driver.on_startup
async def _init() -> None:
    global _llm, _identity_mgr, _timeline, _short_term

    bot_config = load_config()

    long_term = LongTermMemory(memory_dir=bot_config.memory.dir)
    _short_term = ShortTermMemory()
    _timeline = GroupTimeline(max_messages=bot_config.group.max_timeline_messages)
    prompt_builder = PromptBuilder(long_term=long_term)

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
    await _identity_mgr.load_file(bot_config.identity.file)

    _llm = LLMClient(
        base_url=bot_config.llm.base_url,
        api_key=bot_config.llm.api_key,
        model=bot_config.llm.model,
        prompt_builder=prompt_builder,
        short_term=_short_term,
        tools=tools,
        max_context_tokens=bot_config.llm.context.max_context_tokens,
        compact_ratio=bot_config.llm.context.compact_ratio,
        group_timeline=_timeline,
        long_term=long_term,
        warm_enabled=bot_config.llm.cache.warm_enabled,
        warm_interval_messages=bot_config.llm.cache.warm_interval_messages,
        warm_ttl_seconds=bot_config.llm.cache.warm_ttl_seconds,
    )


@driver.on_shutdown
async def _shutdown() -> None:
    await _llm.close()


@driver.on_bot_connect
async def _on_connect(bot: Bot) -> None:
    """Bot 连接后拉取群历史消息，填充时间线。"""
    bot_config = load_config()
    try:
        group_list: list[dict[str, object]] = await bot.get_group_list()
        group_ids = [str(g["group_id"]) for g in group_list]
        logger.info("loading history | groups={}", len(group_ids))
        await load_group_history(
            napcat_url=bot_config.napcat.api_url,
            group_ids=group_ids,
            timeline=_timeline,
            count=bot_config.group.history_load_count,
        )
    except Exception:
        logger.exception("failed to load group history")


def _session_id(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"group_{event.group_id}"
    return f"private_{event.user_id}"


# ── 群聊时间线收集（仅群消息） ──

group_listener = on_message(priority=1, block=False)


@group_listener.handle()
async def collect_group_context(bot: Bot, event: GroupMessageEvent) -> None:
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

    # 缓存预热检查
    identity = _identity_mgr.resolve(_session_id(event), group_id, text)
    _llm.maybe_warm(group_id, identity, str(event.user_id))


# ── /identity 切换人设 ──

identity_cmd = on_command("identity", aliases={"人设"}, priority=5, block=True)


@identity_cmd.handle()
async def handle_identity(event: MessageEvent, args: Message = CommandArg()) -> None:  # noqa: B008
    arg = args.extract_plain_text().strip()
    sid = _session_id(event)

    if not arg or arg == "list":
        names = [f"{'* ' if i.id == 'default' else ''}{i.id} ({i.name})" for i in _identity_mgr.list_identities()]
        await identity_cmd.finish("可用人设:\n" + "\n".join(names))

    if arg == "reset":
        _identity_mgr.clear_override(sid)
        await identity_cmd.finish("已恢复自动匹配人设")

    result = _identity_mgr.switch(sid, arg)
    if result:
        await identity_cmd.finish(f"已切换人设: {result.name}")
    else:
        await identity_cmd.finish(f"未找到人设: {arg}")


# ── 对话 ──

chat = on_message(rule=to_me(), priority=10, block=True)


@chat.handle()
async def handle_chat(bot: Bot, event: MessageEvent) -> None:
    user_text = event.get_plaintext().strip()
    if not user_text:
        return

    sid = _session_id(event)
    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    identity = _identity_mgr.resolve(sid, group_id, user_text)

    ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=group_id)

    try:
        reply = await _llm.chat(
            session_id=sid,
            user_id=str(event.user_id),
            user_text=user_text,
            identity=identity,
            group_id=group_id,
            ctx=ctx,
        )
    except Exception:
        logger.exception("chat error")
        reply = "出错了，请稍后再试"

    await chat.finish(reply)
```

- [ ] **Step 3: 更新 bot.py 支持 CLI 参数**

```python
# bot.py
import argparse
import os

import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

# 解析 CLI 参数，在 NoneBot 初始化前完成
parser = argparse.ArgumentParser(description="QQ Bot")
parser.add_argument("--config", default=None, help="配置文件路径 (默认: config.toml)")
parser.add_argument("--llm-base-url", default=None, help="LLM API base URL")
parser.add_argument("--llm-api-key", default=None, help="LLM API key")
parser.add_argument("--llm-model", default=None, help="LLM model name")
args = parser.parse_args()

# 将 CLI 参数存入环境变量，供 config_loader 读取
if args.config:
    os.environ["BOT_CONFIG_PATH"] = args.config
if args.llm_base_url:
    os.environ["_CLI_LLM_BASE_URL"] = args.llm_base_url
if args.llm_api_key:
    os.environ["_CLI_LLM_API_KEY"] = args.llm_api_key
if args.llm_model:
    os.environ["_CLI_LLM_MODEL"] = args.llm_model

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)

nonebot.load_from_toml("pyproject.toml")

if __name__ == "__main__":
    nonebot.run()
```

- [ ] **Step 4: 更新 config_loader 读取 CLI 环境变量**

在 `src/config_loader.py` 的 `load_config` 函数开头，增加从环境变量读取 CLI 参数的逻辑：

```python
def load_config(
    config_path: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> BotConfig:
    """加载配置，按 TOML → 环境变量 → CLI 逐层覆盖。"""
    data: dict[str, Any] = {}

    # 自动检测配置文件和 CLI 覆盖
    if config_path is None:
        config_path = os.environ.get("BOT_CONFIG_PATH")
        if config_path is None:
            default = Path("config.toml")
            if default.exists():
                config_path = str(default)

    if cli_overrides is None:
        cli_overrides = {}
        for env_suffix, cli_key in [
            ("LLM_BASE_URL", "llm_base_url"),
            ("LLM_API_KEY", "llm_api_key"),
            ("LLM_MODEL", "llm_model"),
        ]:
            val = os.environ.pop(f"_CLI_{env_suffix}", None)
            if val is not None:
                cli_overrides[cli_key] = val

    # ... 其余逻辑不变
```

- [ ] **Step 5: 运行 lint**

Run: `uv run ruff check src/ && uv run ruff check bot.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/plugins/chat/__init__.py src/memory/history_loader.py bot.py src/config_loader.py
git commit -m "feat: wire up GroupTimeline, cache warming, and TOML config in plugin"
```

---

### Task 7: 清理旧代码 + 更新测试

**Files:**
- Delete: `src/memory/group_context.py`
- Delete: `tests/test_group_context.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_prompt.py`

- [ ] **Step 1: 检查 group_context 是否还有引用**

Run: `uv run ruff check src/ 2>&1; grep -r "group_context\|GroupContext" src/ tests/ --include="*.py" | grep -v __pycache__ | grep -v group_timeline`

如果只剩 `group_context.py` 自身和 `test_group_context.py`，可以安全删除。

- [ ] **Step 2: 删除旧文件**

```bash
rm src/memory/group_context.py tests/test_group_context.py
```

- [ ] **Step 3: 清理 conftest.py 中旧 fixture**

从 `tests/conftest.py` 中删除 `GroupContext` 导入和 `group_context` fixture。最终文件：

```python
# tests/conftest.py
from pathlib import Path

import pytest

from src.memory.group_timeline import GroupTimeline
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory


@pytest.fixture
def short_term() -> ShortTermMemory:
    return ShortTermMemory()


@pytest.fixture
def long_term(tmp_path: Path) -> LongTermMemory:
    return LongTermMemory(memory_dir=str(tmp_path))


@pytest.fixture
def group_timeline() -> GroupTimeline:
    return GroupTimeline(max_messages=50)
```

- [ ] **Step 4: 更新 test_prompt.py**

读取 `tests/test_prompt.py`，移除所有对 `GroupContext` 和 `build_context_message` 的引用。`PromptBuilder` 构造函数现在只接受 `long_term` 参数。

- [ ] **Step 5: 运行全部测试**

Run: `uv run pytest -v`
Expected: 所有测试通过（除了可能因 NoneBot 初始化问题跳过的 e2e 测试）

- [ ] **Step 6: 运行 lint + type check**

Run: `uv run ruff check src/ tests/ && uv run pyright`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove GroupContext, update tests for GroupTimeline"
```

---

### Task 8: 集成验证

**Files:** (no new files)

- [ ] **Step 1: 运行全部测试**

Run: `uv run pytest -v`
Expected: 全部通过

- [ ] **Step 2: 运行 lint + type check**

Run: `uv run ruff check src/ tests/ && uv run pyright`
Expected: clean

- [ ] **Step 3: 验证 config.example.toml 可被加载**

```bash
uv run python -c "
from src.config_loader import load_config
cfg = load_config('config.example.toml')
print(f'model={cfg.llm.model}')
print(f'warm_enabled={cfg.llm.cache.warm_enabled}')
print(f'max_timeline={cfg.group.max_timeline_messages}')
print('OK')
"
```
Expected: 输出配置值和 OK

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "chore: integration verification pass"
```
