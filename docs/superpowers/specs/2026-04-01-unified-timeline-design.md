# 群聊统一时间线 + 缓存预热

> 日期: 2026-04-01
> 状态: 设计完成，待实施

## 问题

当前群聊场景有两套独立的消息存储：

- **GroupContext**：旁听群里所有人的消息，最近 50 条，作为文本块注入 messages 开头
- **ShortTermMemory**：仅记录 @bot 的交互轮次，作为 user/assistant 对话历史

这导致两个问题：

1. **Anthropic 缓存命中率极低**：群聊上下文在对话历史之前注入，且每条群消息都会改变其内容。Anthropic prompt cache 是前缀匹配，群上下文一变，后面整个对话历史的缓存全部失效。日志显示 `cache_read` 长期固定在 ~2278 tokens（仅 system blocks），命中率低至 14%。
2. **模型视角割裂**：模型看到的是"旁听笔记 + 自己的对话"两个割裂的信息源，而非连贯的群聊流。

## 方案

### 1. TOML 配置体系

从 `.env` 纯 Pydantic 迁移到 **TOML + CLI 覆盖 + 环境变量覆盖**。

配置文件默认路径为工作目录下的 `config.toml`，可通过 CLI `--config` 指定。

优先级：**CLI 参数 > 环境变量 > TOML 文件 > 默认值**

```toml
# config.example.toml

[llm]
base_url = "http://127.0.0.1:34567/v1"     # 环境变量: LLM_BASE_URL
api_key = "sk-placeholder"                   # 环境变量: LLM_API_KEY (建议用环境变量)
model = "claude-sonnet-4-20250514"           # 环境变量: LLM_MODEL
max_tokens = 1024

[llm.context]
max_context_tokens = 200_000
compact_ratio = 0.7                          # input_tokens > max * ratio 时触发 compact

[llm.cache]
warm_enabled = true                          # 是否启用缓存预热
warm_interval_messages = 10                  # 每积累 N 条新群消息触发一次预热
warm_ttl_seconds = 300                       # 距上次 API 调用超过此秒数不预热

[memory]
dir = "data/memories"

[identity]
file = "identities.md"

[group]
max_timeline_messages = 200                  # 群时间线最大消息数
history_load_count = 30                      # 启动时拉取的历史消息数

[napcat]
api_url = "http://localhost:29300"           # 环境变量: NAPCAT_API_URL

[bot]
superusers = []
```

支持环境变量覆盖的字段在 `config.example.toml` 中以注释标注。

CLI 支持：
- `--config PATH` 指定配置文件路径
- `--llm-base-url`、`--llm-api-key`、`--llm-model` 等常用覆盖

### 2. GroupTimeline —— 统一时间线

新增 `src/memory/group_timeline.py`，替代群聊场景下的 GroupContext + ShortTermMemory。

#### 数据模型

```python
class TimelineMessage:
    role: Literal["user", "assistant"]
    speaker: str | None       # user 时为 "昵称(QQ号)"，assistant 时为 None
    content: str
    is_bot_trigger: bool      # 是否 @bot
```

#### 状态（每个群一个）

```python
class _GroupState:
    messages: list[TimelineMessage]
    summary: str
    last_input_tokens: int
    new_msg_count: int            # 自上次 API 调用后的新消息计数
    last_api_call_time: float     # 上次 API 调用的时间戳
```

#### Anthropic messages 转换

连续的非 bot 消息合并为一个 `user` 块（带 speaker 前缀），bot 回复作为 `assistant` 块：

```
时间线:
  user  小明(10001): 今天天气好
  user  小红(10002): 是啊
  user  小明(10001): @bot 你觉得呢
  assistant: 确实不错~
  user  Daisy(10002): 笑死

转为 Anthropic messages:
  {"role": "user", "content": "小明(10001): 今天天气好\n小红(10002): 是啊\n小明(10001): 你觉得呢"}
  {"role": "assistant", "content": "确实不错~"}
  {"role": "user", "content": "Daisy(10002): 笑死"}
```

#### 与私聊的关系

ShortTermMemory 保留，仅用于私聊。群聊完全由 GroupTimeline 接管。LLMClient.chat() 根据 group_id 是否存在选择数据源。

#### Compact

沿用现有 token 阈值策略：`input_tokens > max_context_tokens * compact_ratio` 时触发。

Compact 增强：修改 compact prompt，让 LLM 同时输出摘要和值得长期记住的用户信息，格式为 JSON：

```json
{"summary": "...", "memories": [{"user_id": "...", "nickname": "...", "traits": {...}, "event": "..."}]}
```

compact 完成后遍历 memories，调用 LongTermMemory 写入。一次 API 调用同时完成压缩和记忆提取。

### 3. 缓存预热

非 @bot 消息到达时，有条件地发起 `max_tokens=1` 的 API 调用以预热缓存前缀。

#### 触发条件（三个 AND）

1. `warm_enabled = true`
2. `new_msg_count >= warm_interval_messages`（默认 10 条）
3. `time.now() - last_api_call_time < warm_ttl_seconds`（默认 300 秒，超过说明缓存已过期）
4. 没有正在进行的预热请求

#### 执行方式

- `asyncio.create_task()` 异步执行，不阻塞消息处理
- 同一时间只允许一个预热请求（bool flag 防并发）
- 使用与正式请求完全相同的 prompt 构建逻辑
- 丢弃返回内容，重置 `new_msg_count`
- 失败静默忽略（log warning）

#### 与 @bot 请求并发

预热是 max_tokens=1，通常几百毫秒完成。如果 @bot 在预热过程中到达：

- @bot 正常发起请求，不等预热
- 预热先完成 → @bot 享受 cache_read
- @bot 先发出 → 预热白做一次，无副作用
- @bot 请求完成后重置计数器

### 4. 整体数据流

```
群消息到达 (NoneBot event)
  │
  ├── group_listener (priority=1, 所有群消息)
  │     → GroupTimeline.add_message(role="user", speaker, content)
  │     → 检查预热条件 → 满足则 asyncio.create_task(_warm_cache())
  │
  └── chat handler (priority=10, 仅 @bot)
        → GroupTimeline.add_message(role="user", speaker, content, is_bot_trigger=True)
        → LLMClient.chat()
        │   ├── compact 检查 → 压缩 + 提取长期记忆
        │   ├── system blocks (人设 + 用户记忆)
        │   ├── messages: [摘要_cached] [时间线...倒数第2条_cached] [最新]
        │   ├── API 调用 (tool loop)
        │   └── GroupTimeline.add_message(role="assistant", content=reply)
        │       重置 new_msg_count / last_api_call_time
        └── 发送回复

私聊消息 → ShortTermMemory (不变)
```

### 5. 模块变更清单

| 文件 | 变更 |
|------|------|
| 新增 `src/memory/group_timeline.py` | GroupTimeline 类 |
| 新增 `src/config_loader.py` | TOML 加载 + CLI + 环境变量覆盖 |
| 新增 `config.example.toml` | 配置示例 |
| 改 `src/config.py` | BotConfig 重构，增加新字段 |
| 改 `src/llm/client.py` | chat() 群聊走 GroupTimeline；新增 _warm_cache()；compact 增加记忆提取 |
| 改 `src/llm/prompt.py` | 删除 build_context_message() |
| 改 `src/plugins/chat/__init__.py` | group_listener 写入 GroupTimeline；初始化走新配置 |
| 改 `src/memory/history_loader.py` | 启动时写入 GroupTimeline |
| 改 `bot.py` | 配置加载改用 config_loader |
| 删 `src/memory/group_context.py` | 被 GroupTimeline 替代 |
| 改 tests/ | 新增 test_group_timeline.py；更新 conftest.py；删除 test_group_context.py |

### 6. 不动的部分

- `src/llm/prompt.py` 的 `build_blocks()`（system blocks 不变）
- `src/memory/long_term.py`（只是被 compact 多调用一下）
- `src/memory/short_term.py`（保留，私聊专用）
- `src/tools/*`（所有工具不变）
- `src/identity/*`（身份系统不变）

### 7. 缓存效果预期

改造前：
- `cache_read` 固定 ~2278 tokens（仅 system blocks）
- 命中率 14-30%，大部分 token 是 cache_create

改造后：
- 时间线作为稳定前缀，两次 @bot 之间的历史全部 cache_read
- 预热进一步减少 @bot 时的 cache_create 量
- 预期命中率 60-90%（取决于 @bot 频率和群活跃度）
- @bot 响应延迟显著降低（减少 cache_create 的处理时间）
