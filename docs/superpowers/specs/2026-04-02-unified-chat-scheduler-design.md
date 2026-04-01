# Unified Chat Scheduler — 统一群聊调度

**Date:** 2026-04-02
**Goal:** 将 ProactiveEvaluator（廉价模型决策）+ warm_cache（空调用预热）+ chat()（主模型回复）三条独立路径合并为一条：所有群消息都走主模型，模型自己决定回不回复。副作用是每次调用都刷新 prompt cache。

## 动机

当前架构有三套独立的 API 调用路径：

1. **ProactiveEvaluator** — 廉价模型 + 128 max_tokens + 最近 20 行上下文，batch/debounce 后判断是否插话
2. **warm_cache** — 主模型 + max_tokens=1，空调用刷新 prompt cache
3. **chat()** — 主模型，正式对话（@bot 触发 或 proactive 决定回复后触发）

问题：
- 决定回复时需要两次 API 调用（eval + chat），浪费
- warm_cache 是纯开销
- ProactiveEvaluator 376 行状态机（batch/debounce/pending/cooldown），复杂度高
- 廉价模型只看 20 行上下文，决策质量有限

## 设计

### 核心思路

不再区分"评估"和"回复"阶段。每次 debounce 触发后直接走主模型 chat()，注入 `pass_turn` 工具。模型要么输出文本（发消息），要么调 `pass_turn`（不发，记日志）。每次调用都是一次 cache warm。

### 消息流

```
群消息(非@) → timeline.add()
             → GroupChatScheduler.notify()
               → debounce (N秒) / batch (M条)
               → LLMClient.chat(allow_skip=True)
               → pass_turn? → 不发，记日志
               → 有文本?   → 发群消息，timeline.add(assistant)

群消息(@bot) → timeline.add()
              → GroupChatScheduler.interrupt()  ← 取消该群 debounce + 正在跑的 task
              → LLMClient.chat(allow_skip=False)  ← 同步 await，必须回复
              → 发群消息

私聊 → (不变) → LLMClient.chat() → 回复
```

### GroupChatScheduler (`src/llm/scheduler.py`, ~150 行)

```python
class _GroupSlot:
    debounce_task: asyncio.Task | None    # 当前 debounce 定时器
    running_task: asyncio.Task | None     # 正在跑的 chat() 调用
    msg_count: int                        # debounce 期间累积的消息数

class GroupChatScheduler:
    def __init__(
        self,
        llm: LLMClient,
        timeline: GroupTimeline,
        identity_mgr: IdentityManager,
        debounce_seconds: float = 5.0,
        batch_size: int = 10,
    ): ...
```

**`notify(group_id)`** — 普通群消息调用：

1. `identity = identity_mgr.resolve()`；如果 `identity.proactive is None` → return（无 proactive 规则的 identity 不触发 debounce）
2. `msg_count++`
3. 如果 `running_task` 正在跑 → 什么都不做（消息已在 timeline，下次会看到）
4. 取消旧 `debounce_task`
5. `msg_count >= batch_size` → 立即 `_fire()`
6. 否则 → 启动新 `debounce_task`（sleep N 秒后 `_fire()`）

**`interrupt(group_id)`** — @bot 时调用：

1. 取消 `debounce_task`（如有）
2. 取消 `running_task`（如有）
3. 返回（不调 chat，@bot handler 自己同步 await）

**`_fire(group_id)`** — debounce/batch 触发：

1. `msg_count = 0`
2. `running_task = create_task(_do_chat(group_id))`

**`_do_chat(group_id)`**：

1. 构造参数：`session_id = f"group_{group_id}"`，`user_id = ""`，`user_text = ""`，`identity = identity_mgr.resolve()`，`ctx = ToolContext(bot=bot, user_id="", group_id=group_id)`
2. `on_segment` = 通过 `bot.send_group_msg` 发送分段消息
3. `reply = await llm.chat(..., allow_skip=True)`
4. `reply is not None` → 发群消息（通过 bot 实例）
5. `running_task = None`
6. 如果被 `CancelledError` 中断（@bot 抢占），直接退出。已发的 segments 留在群里不撤回——属于正常对话上下文，@bot 回复自然跟在后面

**`set_bot(bot)`** — bot connect 后注入 Bot 实例，用于发群消息。

**`close()`** — shutdown 时取消所有 tasks。

### LLMClient 改动

**签名变化：**

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
    allow_skip: bool = False,  # 新增
) -> str | None:               # None = pass_turn
```

**pass_turn 内置工具：**

```python
PASS_TURN_TOOL = {
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

- `allow_skip=True` 时追加到工具列表末尾
- tool loop 中拦截：模型调 `pass_turn` → 记日志 + `set_input_tokens`（保留 cache 效果）→ 返回 `None`
- 如果模型同时输出文本和调 `pass_turn`，以 `pass_turn` 为准，丢弃文本
- 不注册到 ToolRegistry，不需要 `execute()` 实现

**删除：**
- `proactive_hint` 参数及注入逻辑
- `_warm_cache()` / `maybe_warm()` 方法
- `_warming` 状态
- 构造函数中 `warm_enabled` / `warm_interval_messages` / `warm_ttl_seconds` 参数

### Identity & System Prompt

`Identity.proactive` 字段保留，语义从"廉价模型决策 prompt"变为"主模型群聊指令"。

注入位置：`PromptBuilder.build_blocks()` 中，群聊时追加到层 1 末尾：

```python
if group_id and identity.proactive:
    base_text += "\n\n" + identity.proactive
```

### Config 变化

**删除：**
- `CacheConfig` 类
- `ProactiveConfig` 类
- `LLMConfig.cache` 字段
- `BotConfig.proactive` 字段

**新增（GroupConfig）：**

```python
class GroupConfig(BaseModel):
    max_timeline_messages: int = 200
    history_load_count: int = 30
    allowed_groups: list[int] = []
    debounce_seconds: float = 5.0   # 新增
    batch_size: int = 10            # 新增
```

### GroupTimeline 改动

**删除：**
- `should_warm()` / `reset_warm_counter()` 方法
- `_GroupState.new_msg_count` 字段
- `_GroupState.last_api_call_time` 字段
- `add()` 中 `new_msg_count += 1`

**保留：** `set_input_tokens` / `get_input_tokens` / `needs_compact` / `compact`（compact 机制不变）。

`set_input_tokens` 简化为只记录 token 数（不再写 `last_api_call_time` 和清零 `new_msg_count`）。

### 插件层改动 (`plugins/chat/__init__.py`)

**全局变量：** `_proactive` → `_scheduler`

**`_init()`：** 删 ProactiveEvaluator 初始化 + LLMClient warm_* 参数，新增 GroupChatScheduler 初始化。

**`_on_connect()`：** `_scheduler.set_bot(bot)` 注入 bot 实例。删除历史消息触发 proactive 检测（126-167 行）。

**`collect_group_context()`：** 删 `maybe_warm` + `_proactive.notify`，替换为 `_scheduler.notify(group_id)`。

**`handle_chat()`：** 群聊时先调 `_scheduler.interrupt(group_id)` 清场，然后同步 await chat()。删 `proactive_hint` 参数。

**`_shutdown()`：** `_proactive.close()` → `_scheduler.close()`。

### config_loader.py 改动

删除 `_ENV_MAP` 中 `"PROACTIVE_MODEL": "proactive.model"` 条目。

## 并发控制

- **@bot 抢占：** `interrupt()` 取消 debounce + 正在跑的 task，@bot handler 独占 chat()
- **无冷却期：** 完全信任模型判断，不限制主动回复频率
- **debounce 期间有 running_task：** notify 不做任何事，消息已在 timeline

## 文件变更

| 操作 | 文件 |
|---|---|
| 删除 | `src/llm/proactive.py` |
| 新增 | `src/llm/scheduler.py` |
| 修改 | `src/llm/client.py` |
| 修改 | `src/plugins/chat/__init__.py` |
| 修改 | `src/config.py` |
| 修改 | `src/config_loader.py` |
| 修改 | `src/memory/group_timeline.py` |
| 修改 | `src/llm/prompt.py` |
| 修改 | `config.toml`（如有） |
| 删除 | `tests/test_proactive.py` |
| 新增 | `tests/test_scheduler.py` |
| 修改 | `tests/test_group_timeline.py` — 删 4 个 warm 相关测试 |
| 修改 | `tests/test_config_loader.py` — 删 proactive/cache config 断言 |

**净效果：** 删 ~430 行，加 ~150 行，净减 ~280 行。
