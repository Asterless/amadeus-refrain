# 主动插话功能 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 bot 根据 LLM 判断自主加入群聊对话，无需被 @。

**Architecture:** 在 `group_listener` 收集消息后，用可配置的廉价模型（默认 Haiku）对最近上下文做 YES/NO 决策。决策通过则走完整 `_llm.chat()` 流程生成回复。每个 identity 可配置独立的 `proactive` 规则字符串，没有该字段则不会主动插话。per-group asyncio.Lock 防并发，严格 timeout 防阻塞。

**Tech Stack:** Python 3.12, aiohttp, NoneBot2, Anthropic Messages API

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/identity/models.py` | Modify | Identity 加 `proactive` 字段 |
| `src/identity/manager.py` | Modify | `_parse_markdown` 解析 `proactive` 元数据 |
| `src/config.py` | Modify | 新增 `ProactiveConfig` |
| `src/config_loader.py` | Modify | 新增环境变量映射 |
| `src/llm/proactive.py` | Create | `ProactiveEvaluator` 类：决策调用 + 冷却 + 锁 |
| `src/plugins/chat/__init__.py` | Modify | 集成 `ProactiveEvaluator` |
| `soul/identities.md` | Modify | default 人设加 `proactive` 规则 |
| `tests/test_identity.py` | Modify | 测试 proactive 字段解析 |
| `tests/test_proactive.py` | Create | `ProactiveEvaluator` 单元测试 |

---

### Task 1: Identity model 加 proactive 字段

**Files:**
- Modify: `src/identity/models.py:13-21`
- Modify: `src/identity/manager.py:26-60`
- Modify: `tests/test_identity.py`

- [ ] **Step 1: 写 proactive 字段解析的失败测试**

在 `tests/test_identity.py` 末尾追加：

```python
def test_parse_markdown_with_proactive() -> None:
    md = """\
## bot
- name: Bot
- proactive: 只在有人求助时插话。回答 YES 或 NO。

Bot 人设。
"""
    identities = _parse_markdown(md)
    assert identities[0].proactive == "只在有人求助时插话。回答 YES 或 NO。"


def test_parse_markdown_without_proactive() -> None:
    md = """\
## bot
- name: Bot

Bot 人设。
"""
    identities = _parse_markdown(md)
    assert identities[0].proactive is None
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_identity.py::test_parse_markdown_with_proactive tests/test_identity.py::test_parse_markdown_without_proactive -v`
Expected: FAIL — `Identity` 没有 `proactive` 字段

- [ ] **Step 3: Identity model 加字段**

在 `src/identity/models.py` 的 `Identity` 类中加：

```python
proactive: str | None = Field(default=None, description="主动插话判断规则，None 表示不主动插话")
```

- [ ] **Step 4: _parse_markdown 解析 proactive**

在 `src/identity/manager.py` 的 `_parse_markdown` 函数中，`identities.append(...)` 之前，从 `meta` 提取 proactive：

```python
proactive = meta.get("proactive")
```

并在 `Identity(...)` 构造中加 `proactive=proactive`：

```python
identities.append(
    Identity(
        id=identity_id,
        name=meta.get("name", identity_id),
        personality="\n".join(personality_lines).strip(),
        trigger=TriggerRule(groups=groups, keywords=keywords),
        priority=int(meta.get("priority", "0")),
        proactive=proactive,
    )
)
```

- [ ] **Step 5: 运行测试验证通过**

Run: `uv run pytest tests/test_identity.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/identity/models.py src/identity/manager.py tests/test_identity.py
git commit -m "feat: add proactive field to Identity model"
```

---

### Task 2: ProactiveConfig 配置

**Files:**
- Modify: `src/config.py:50-74`
- Modify: `src/config_loader.py:13-18`
- Modify: `tests/test_config_loader.py`

- [ ] **Step 1: 写配置加载的失败测试**

在 `tests/test_config_loader.py` 追加：

```python
def test_proactive_config_defaults() -> None:
    from src.config import BotConfig
    config = BotConfig()
    assert config.proactive.model == "claude-haiku-4-5-20251001"
    assert config.proactive.timeout == 3.0
    assert config.proactive.context_lines == 20
    assert config.proactive.cooldown == 60


def test_proactive_config_from_toml(tmp_path: Path) -> None:
    from src.config_loader import load_config
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[proactive]\nmodel = "custom-model"\ntimeout = 5.0\ncooldown = 30\n'
    )
    config = load_config(config_path=str(toml_file))
    assert config.proactive.model == "custom-model"
    assert config.proactive.timeout == 5.0
    assert config.proactive.cooldown == 30
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_config_loader.py::test_proactive_config_defaults tests/test_config_loader.py::test_proactive_config_from_toml -v`
Expected: FAIL — `BotConfig` 没有 `proactive` 字段

- [ ] **Step 3: 添加 ProactiveConfig**

在 `src/config.py` 中 `GroupConfig` 之前加：

```python
class ProactiveConfig(BaseModel):
    """主动插话配置。"""

    model: str = "claude-haiku-4-5-20251001"
    timeout: float = 3.0  # 决策调用超时（秒）
    context_lines: int = 20  # 传给决策模型的最近消息条数
    cooldown: int = 60  # 主动插话后冷却时间（秒）
```

在 `BotConfig` 中加：

```python
proactive: ProactiveConfig = ProactiveConfig()
```

- [ ] **Step 4: config_loader 加环境变量映射**

在 `src/config_loader.py` 的 `_ENV_MAP` 中加：

```python
"PROACTIVE_MODEL": "proactive.model",
```

- [ ] **Step 5: 运行测试验证通过**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/config.py src/config_loader.py tests/test_config_loader.py
git commit -m "feat: add ProactiveConfig for autonomous chat"
```

---

### Task 3: ProactiveEvaluator 核心逻辑

**Files:**
- Create: `src/llm/proactive.py`
- Create: `tests/test_proactive.py`

- [ ] **Step 1: 写 ProactiveEvaluator 的失败测试**

创建 `tests/test_proactive.py`：

```python
"""ProactiveEvaluator 单元测试。"""

import asyncio
import time

import pytest

from src.identity.models import Identity
from src.llm.proactive import ProactiveEvaluator
from src.memory.group_timeline import GroupTimeline


def _make_identity(proactive: str | None = None) -> Identity:
    return Identity(
        id="test",
        name="测试",
        personality="测试人设",
        proactive=proactive,
    )


@pytest.fixture
def timeline() -> GroupTimeline:
    tl = GroupTimeline(max_messages=50)
    for i in range(5):
        tl.add("g1", role="user", speaker=f"用户{i}(100{i})", content=f"消息{i}")
    return tl


class TestShouldEvaluate:
    def test_no_proactive_rule(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u")
        identity = _make_identity(proactive=None)
        assert ev.should_evaluate("g1", identity) is False

    def test_has_proactive_rule(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u")
        identity = _make_identity(proactive="随便插话")
        assert ev.should_evaluate("g1", identity) is True

    def test_cooldown_blocks(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", cooldown=60)
        identity = _make_identity(proactive="随便插话")
        ev._last_proactive["g1"] = time.monotonic()  # 刚刚插话过
        assert ev.should_evaluate("g1", identity) is False

    def test_cooldown_expired(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", cooldown=60)
        identity = _make_identity(proactive="随便插话")
        ev._last_proactive["g1"] = time.monotonic() - 120  # 很久以前
        assert ev.should_evaluate("g1", identity) is True

    def test_locked_group_blocks(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u")
        identity = _make_identity(proactive="随便插话")
        ev._locks["g1"] = asyncio.Lock()
        ev._locks["g1"]._locked = True  # 模拟锁被持有
        # locked() 为 True 时 should_evaluate 返回 False
        assert ev.should_evaluate("g1", identity) is False


class TestBuildDecisionPrompt:
    def test_prompt_content(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", context_lines=3)
        identity = _make_identity(proactive="只在有人求助时插话")
        system, messages = ev.build_decision_prompt("g1", identity)
        assert "只在有人求助时插话" in system[0]["text"]
        # messages 的 user content 应包含最近 3 条消息
        user_text = messages[0]["content"]
        assert "消息2" in user_text
        assert "消息3" in user_text
        assert "消息4" in user_text

    def test_prompt_respects_context_lines(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", context_lines=2)
        identity = _make_identity(proactive="规则")
        _, messages = ev.build_decision_prompt("g1", identity)
        user_text = messages[0]["content"]
        # 只取最近 2 条
        assert "消息3" in user_text
        assert "消息4" in user_text
        assert "消息0" not in user_text
```

- [ ] **Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_proactive.py -v`
Expected: FAIL — `src.llm.proactive` 不存在

- [ ] **Step 3: 实现 ProactiveEvaluator**

创建 `src/llm/proactive.py`：

```python
"""主动插话评估器：用廉价模型判断是否应主动加入群聊。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
from loguru import logger

from src.identity.models import Identity
from src.memory.group_timeline import GroupTimeline


class ProactiveEvaluator:
    """评估是否应主动插话，并在决定插话时调用回调。"""

    def __init__(
        self,
        *,
        timeline: GroupTimeline,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 3.0,
        context_lines: int = 20,
        cooldown: int = 60,
    ) -> None:
        self._timeline = timeline
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._context_lines = context_lines
        self._cooldown = cooldown

        self._locks: dict[str, asyncio.Lock] = {}
        self._last_proactive: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 前置检查
    # ------------------------------------------------------------------

    def should_evaluate(self, group_id: str, identity: Identity) -> bool:
        """快速判断是否需要进行决策调用。"""
        if not identity.proactive:
            return False

        # 冷却期内不评估
        last = self._last_proactive.get(group_id, 0.0)
        if time.monotonic() - last < self._cooldown:
            return False

        # 已有评估/回复在进行中
        lock = self._locks.get(group_id)
        if lock and lock.locked():
            return False

        return True

    # ------------------------------------------------------------------
    # 决策 prompt 构建
    # ------------------------------------------------------------------

    def build_decision_prompt(
        self, group_id: str, identity: Identity,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """构建决策用的 system blocks 和 messages。"""
        messages = self._timeline.get_messages(group_id)
        recent = messages[-self._context_lines :]

        lines: list[str] = []
        for msg in recent:
            if msg["role"] == "assistant":
                lines.append(f"{identity.name}: {msg['content']}")
            elif msg["speaker"]:
                lines.append(f"{msg['speaker']}: {msg['content']}")
            else:
                lines.append(msg["content"])

        system = [{"type": "text", "text": identity.proactive}]
        user_messages = [{"role": "user", "content": "\n".join(lines)}]
        return system, user_messages

    # ------------------------------------------------------------------
    # 决策调用
    # ------------------------------------------------------------------

    async def evaluate(self, group_id: str, identity: Identity) -> bool:
        """调用廉价模型判断是否应插话。返回 True 表示应插话。"""
        lock = self._locks.setdefault(group_id, asyncio.Lock())

        async with lock:
            system, messages = self.build_decision_prompt(group_id, identity)

            timeout = aiohttp.ClientTimeout(total=self._timeout)
            body = {
                "model": self._model,
                "system": system,
                "messages": messages,
                "max_tokens": 8,
                "stream": False,
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2024-10-22",
            }

            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self._base_url}/v1/messages", json=body, headers=headers,
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        text = ""
                        for block in data.get("content", []):
                            if block.get("type") == "text":
                                text += block.get("text", "")
                        decision = text.strip().upper().startswith("YES")
                        logger.info(
                            "proactive eval | group={} decision={} raw={!r}",
                            group_id, decision, text.strip()[:50],
                        )
                        if decision:
                            self._last_proactive[group_id] = time.monotonic()
                        return decision
            except asyncio.TimeoutError:
                logger.warning("proactive eval timeout | group={}", group_id)
                return False
            except Exception:
                logger.warning("proactive eval error | group={}", group_id, exc_info=True)
                return False

    def record_proactive(self, group_id: str) -> None:
        """手动记录插话时间（用于外部调用成功后）。"""
        self._last_proactive[group_id] = time.monotonic()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_proactive.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/proactive.py tests/test_proactive.py
git commit -m "feat: add ProactiveEvaluator for autonomous chat decisions"
```

---

### Task 4: 集成到 chat 插件

**Files:**
- Modify: `src/plugins/chat/__init__.py`

- [ ] **Step 1: 在 _init() 中初始化 ProactiveEvaluator**

在 `src/plugins/chat/__init__.py` 顶部 import 区加：

```python
from src.llm.proactive import ProactiveEvaluator
```

在模块级变量区加：

```python
_proactive: ProactiveEvaluator
```

在 `_init()` 函数中，`_llm = LLMClient(...)` 之后加：

```python
_proactive = ProactiveEvaluator(
    timeline=_timeline,
    model=bot_config.proactive.model,
    api_key=bot_config.llm.api_key,
    base_url=bot_config.llm.base_url,
    timeout=bot_config.proactive.timeout,
    context_lines=bot_config.proactive.context_lines,
    cooldown=bot_config.proactive.cooldown,
)
```

- [ ] **Step 2: 修改 collect_group_context 加入决策逻辑**

将 `collect_group_context` 改为：

```python
@group_listener.handle()
async def collect_group_context(bot: Bot, event: GroupMessageEvent) -> None:
    if _allowed_groups and event.group_id not in _allowed_groups:
        return
    text = event.get_plaintext().strip()
    if not text:
        return

    # 被 @ 的消息交给 chat handler，不走主动插话
    if event.is_tome():
        return

    nickname = event.sender.nickname or str(event.user_id)
    group_id = str(event.group_id)
    _timeline.add(
        group_id,
        role="user",
        speaker=f"{nickname}({event.user_id})",
        content=text,
    )

    identity = _identity_mgr.resolve(_session_id(event), group_id, text)
    _llm.maybe_warm(group_id, identity, str(event.user_id))

    # 主动插话评估
    if not _proactive.should_evaluate(group_id, identity):
        return

    should_reply = await _proactive.evaluate(group_id, identity)
    if not should_reply:
        return

    sid = _session_id(event)
    ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=group_id)

    async def send_segment(text: str) -> None:
        await bot.send_group_msg(group_id=event.group_id, message=text)

    try:
        reply = await _llm.chat(
            session_id=sid,
            user_id=str(event.user_id),
            user_text=text,
            identity=identity,
            group_id=group_id,
            ctx=ctx,
            on_segment=send_segment,
        )
    except Exception:
        logger.exception("proactive chat error | group={}", group_id)
        return

    if reply:
        await bot.send_group_msg(group_id=event.group_id, message=reply)
```

- [ ] **Step 3: 运行 lint 和类型检查**

Run: `uv run ruff check src/plugins/chat/__init__.py`
Run: `uv run pyright src/plugins/chat/__init__.py`
Fix any issues.

- [ ] **Step 4: Commit**

```bash
git add src/plugins/chat/__init__.py
git commit -m "feat: integrate proactive evaluator into group listener"
```

---

### Task 5: Soul 配置 & 端到端验证

**Files:**
- Modify: `soul/identities.md`

- [ ] **Step 1: 给 default 人设加 proactive 规则**

在 `soul/identities.md` 的 `## default` 元数据区加 `proactive` 字段（在 `- name:` 之后）：

```markdown
## default

- name: 牧濑红莉栖
- proactive: |
    你是 Amadeus（牧濑红莉栖），正在观察一个 QQ 群的聊天记录。
    判断你是否应该主动发言。只回答 YES 或 NO，不要解释。
    以下情况可以插话：
    - 有人在讨论你擅长的领域（物理、科学、时间旅行、命运石之门相关）
    - 有人明显在求助或提问，且你能提供有价值的回答
    - 对话中出现了你有强烈看法的话题
    - 有人提到了你（红莉栖、Amadeus、助手、克里斯蒂娜）但没有 @
    以下情况不要插话：
    - 日常寒暄、无意义水群
    - 你没有独特见解可以贡献
    - 群里最近已经有你的发言
```

- [ ] **Step 2: 运行全量测试**

Run: `uv run pytest -v`
Run: `uv run ruff check src/`
Run: `uv run pyright`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add soul/identities.md
git commit -m "feat: add proactive rules to default identity"
```

---

### Task 6: 全量验证 & 收尾

- [ ] **Step 1: 运行全量测试**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 2: Lint + Type check**

Run: `uv run ruff check src/`
Run: `uv run pyright`
Expected: 无错误

- [ ] **Step 3: 最终 commit（如有未提交的修复）**

```bash
git add -A
git commit -m "chore: final fixes for proactive chat feature"
```
