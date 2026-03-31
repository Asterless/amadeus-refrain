# 对话历史 Compact 设计

## 问题

当前 `ShortTermMemory` 用 `deque(maxlen=40)` 管理对话历史。满了之后每轮都丢弃最旧消息，破坏 messages 前缀，导致 Anthropic prompt cache 在 messages 部分完全失效。

## 方案

去掉 deque 硬截断，改为无限累积 + 按需压缩：

- 对话历史用 `list` 无限累积
- 每次 API 调用后，记录返回的 `input_tokens`
- 下次请求前，若 `input_tokens > model_max * 0.7`，触发 compact
- Compact 把历史前半部分用 LLM 压缩成摘要，保留后半部分原始消息
- 群聊上下文放在 compact 摘要之后；首次无摘要时不注入群聊上下文到 messages

## 改动文件

### 1. `src/config.py`

新增字段：

```python
llm_max_context_tokens: int = 200_000   # 模型 context window
compact_ratio: float = 0.7              # 触发 compact 的比例
```

去掉 `short_term_max_rounds`。

### 2. `src/memory/short_term.py`

```python
class SessionState(TypedDict):
    messages: list[ChatMessage]
    summary: str              # compact 摘要，空串=无
    last_input_tokens: int    # 上次 API 返回的 input_tokens

class ShortTermMemory:
    _store: dict[str, SessionState]

    add(session_id, role, content)        # 追加到 messages list
    get(session_id) -> list[ChatMessage]  # 返回 messages
    get_summary(session_id) -> str        # 返回摘要
    set_input_tokens(session_id, n)       # 记录 token 数
    get_input_tokens(session_id) -> int   # 读取 token 数
    needs_compact(session_id, max_tokens, ratio) -> bool
    compact(session_id, split, new_summary)
        # summary = new_summary
        # messages = messages[split:]
    clear(session_id)
```

仍保留 `_MAX_SESSIONS = 500` 的会话数上限。

### 3. `src/llm/client.py`

核心改动在 `chat()` 方法：

```
async def chat(...):
    # 1. compact 检查
    if self._short_term.needs_compact(session_id, max_tokens, ratio):
        await self._compact(session_id)

    # 2. 拼装 messages
    messages = []

    # 摘要（稳定前缀）
    summary = self._short_term.get_summary(session_id)
    if summary:
        messages.append(user: cached_text("[对话摘要]\n{summary}"))
        messages.append(assistant: "好的，我已了解之前的对话内容。")

        # 群聊上下文（仅在有摘要时注入 messages）
        if group_id:
            ctx_text = ...
            messages.append(user: cached_text("[群聊上下文]\n{ctx_text}"))
            messages.append(assistant: "好的，我已了解最近的群聊内容。")

    # 对话历史（完整，不截断）
    history[-2] 加 cache_control

    # 3. API 调用 + 记录 token 数
    result = await self._call(...)
    self._short_term.set_input_tokens(session_id, result["input_tokens"])
```

`_compact()` 方法：

```
async def _compact(session_id):
    history = self._short_term.get(session_id)
    old_summary = self._short_term.get_summary(session_id)
    split = len(history) // 2

    # 用同一模型调 API 生成摘要
    prompt = 压缩指令 + old_summary(如有) + history[:split]
    result = await _call_api(system=[压缩指令], messages=[prompt])
    self._short_term.compact(session_id, split, result["text"])
```

### 4. `_call_api` 返回值

补充 `input_tokens`：

```python
return {"text": ..., "tool_uses": ..., "input_tokens": usage.get("input_tokens", 0)}
```

### 5. `src/plugins/chat/__init__.py`

- `ShortTermMemory()` 不再传 `max_rounds`
- `LLMClient` 构造函数新增 `max_context_tokens` 和 `compact_ratio`

### 6. `src/memory/history_loader.py`

不变。启动时仅加载群聊上下文，`GroupContext.max_messages=50` 限制条数。

## Cache 效果

```
compact 之间（大部分时间）：
  system [人设✓ 记忆✓] | tools [✓] | [摘要✓] [群聊✓] [msg1..msg_n-2✓] [msg_n-1 新]
                                      全部是稳定前缀，cache 命中

compact 发生时（偶尔）：
  摘要更新 → 前缀变化 → 一次 cache miss → 立刻开始新的 cache 积累
```

## 不变的部分

- `group_context.py` — 群聊上下文仍用 deque(50)，旁路记录，与对话历史无关
- `history_loader.py` — 启动时仅加载群聊上下文
- `long_term.py` — 长期记忆不变
- `prompt.py` — system blocks 构建不变，`build_context_message` 调用位置移到 client 中
