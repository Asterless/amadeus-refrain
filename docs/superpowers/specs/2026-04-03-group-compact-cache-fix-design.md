# 群聊 Compact 优化：修复 prompt cache 命中率

## 问题

活跃群聊中 prompt cache 命中率持续低于 25%。

**根因**：`GroupTimeline.add()` 在 buffer 满（max_timeline_messages=200）时从队头淘汰消息。每次淘汰改变第一条 Anthropic 消息内容（合并后的 user batch），破坏从该位置到 messages cache 断点的整个前缀。Anthropic prompt cache 要求前缀逐字节匹配，因此消息区域的 cache 断点④永远无法命中。

**现状数据**：cache_r≈5115（仅 tools+system），cache_w≈20k（全部消息），hit≈20%。

## 方案

删除 `max_timeline_messages` 硬截断。改为按上下文窗口占比触发 LLM compact。compact 产生的摘要（compact block）放在 system 之后、正常消息之前，最多一个，每次替换。

这复用现有 `GroupState.summary` 字段和 `_compact_group` LLM 压缩逻辑，不引入新抽象。

## 改动清单

### 1. `src/config.py`

**CompactConfig** 简化为单阈值：

```python
class CompactConfig(BaseModel):
    ratio: float = 0.7            # input_tokens > max_context_tokens * ratio 时触发
    compress_ratio: float = 0.5   # 压缩前 50% 的消息
    max_failures: int = 3
    cache_hit_warn: float = 90.0
```

- 删除 `micro_ratio`、`full_ratio` 及 `_check_ratios` validator
- 新增 `compress_ratio`

**GroupConfig**：删除 `max_timeline_messages` 字段。

### 2. `src/memory/group_timeline.py`

**`_GroupState.__init__`**：删除 `_max` / `max_messages` 参数。

**`GroupTimeline.__init__`**：删除 `max_messages` 参数。

**`GroupTimeline.add()`**：删除 `if len > max` 截断逻辑。消息无限累积，由 compact 控制大小。

**`GroupTimeline.compact()`**：保持现有逻辑（切前 N 条 + 替换 summary + 重置 token/cache index）。`split` 参数由调用方基于 `compress_ratio` 计算。

**`GroupTimeline.drop_oldest()`**：保留（compact 内部使用）。

其余方法不变。

### 3. `src/llm/client.py`

**`LLMClient.__init__`**：
- 删除 `micro_ratio` 参数
- `full_ratio` 改名为 `compact_ratio`（语义更准确）
- 新增 `compress_ratio: float = 0.5`

**`chat()`** compact 触发逻辑：

```python
# 旧：双阈值 micro + full
# 新：单阈值
if self._timeline.needs_compact(group_id, self._max_context_tokens, self._compact_ratio):
    await self._compact_group(group_id, identity)
```

**`_compact_group()`**：
- `split` 计算改用 `compress_ratio`：`split = max(2, int(len(messages) * self._compress_ratio))`
- 其余逻辑（LLM prompt、memo 提取、JSON 解析、circuit breaker、usage tracking）不变

**删除**：`_micro_compact_group()`

**私聊同步改动**：
- `chat()` 中私聊分支同样改为单阈值触发 `_compact()`
- 删除 `_micro_compact_private()`
- `ShortTermMemory`：如有类似 max 截断逻辑也一并删除

### 4. `src/plugins/chat/__init__.py`

- `GroupTimeline()` 构造：删除 `max_messages=` 参数
- `LLMClient()` 构造：`micro_ratio=` → 删除，`full_ratio=` → `compact_ratio=`，新增 `compress_ratio=`

### 5. `config.example.toml`（如涉及）

更新 `[compact]` 段：删除 `micro_ratio` / `full_ratio`，新增 `ratio` / `compress_ratio`。

## 缓存效果

改动后的 cache 断点布局：

```
① tools[-1]                          ← 全局稳定
② system block 1 (人格+指令)         ← 全局稳定
③ system block 2 (索引+备忘)         ← 准稳定（memo 变更时失效）
   compact block (summary user+ack)   ← compact 间隔内稳定（已有 _cached_text）
④ messages[cached_idx]               ← 移动窗口，前缀现在稳定
```

关键改善：
- 无队头淘汰 → messages 前缀不再被破坏 → ④ 可以命中
- compact block 在两次 compact 之间不变 → 形成稳定锚点
- 预期命中率从 ~20% 提升到 ~90%+

如果 Anthropic 报 breakpoint 超限（summary 的 `_cached_text` 是第 5 个断点），合并 ②③ 为一个 system block。

## 不改动的部分

- `_compact_group` 的 LLM prompt、memo 提取、JSON 解析逻辑
- `_build_group_messages` 中 summary 注入和 cached_idx 逻辑
- `UsageTracker` 的 cache hit 告警
- Dream 系统
- 私聊 `_compact()` 的 LLM prompt（仅改触发逻辑）

## 测试策略

- 现有 compact 相关单元测试更新（删除 max_messages 相关用例，新增 ratio 触发用例）
- `GroupTimeline.add()` 不再截断的验证
- `CompactConfig` 新字段的 validation
- 部署后观察 cache hit 告警是否消失
