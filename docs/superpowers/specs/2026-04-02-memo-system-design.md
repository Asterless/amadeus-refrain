# Memo System Design

Replace the current `.qmd` long-term memory with a markdown-based memo system featuring an index + detail file architecture, @/# cross-reference conventions, in-memory caching, and a Dream agent for periodic consolidation.

## 1. Storage Layer

### File Structure

```
storage/memories/
  index.md                         # Global index, < 200 lines
  users/
    123456.md                      # User memo (prose)
    123456.log                     # Changelog (append-only, debug only)
  groups/
    987654.md                      # Group memo (prose)
    987654.log                     # Changelog
```

### index.md

Lightweight registry. One line per entity, < 150 chars each. Derived from memo first lines — never the source of truth.

```markdown
# users
- @123456 小明 | 杭州·后端·Go/Python | #987654 #111222
- @789012 小红 | 前端·Rust爱好者 | #987654

# groups
- #987654 技术吹水群 | ~10人 轻松玩梗 | @123456 @789012 @345678
```

### Memo Format

#### Metadata (HTML comment, first line)

```
<!-- updated: YYYY-MM-DD HH:MM | source: {source} -->
```

Source values: `compact:private:{session_id}`, `compact:group:{group_id}`, `tool:{session_id}`, `dream`.

#### Identity Principle

**QQ号 is the sole identity identifier.** Nicknames are untrusted — they can be changed at will, can collide across users, and may be deliberately spoofed to impersonate others. All storage, indexing, cross-referencing, and lookup MUST key on QQ号, never on nickname alone.

Nicknames are stored only as a display hint alongside the QQ号, never as an identifier.

#### Tagging Convention

| Tag | Meaning | Format | Example |
|-----|---------|--------|---------|
| `@` | User reference | `@{QQ号}` or `@{QQ号}({nickname})` | `@123456`, `@123456(小明)` |
| `#` | Group reference | `#{group_id}` | `#987654` |

The QQ号 is always required. Nickname in parentheses is optional context — first mention uses `@123456(小明)`, subsequent can use `@123456` alone. Never write a nickname without its QQ号.

#### User Memo Structure (~300 chars)

```
Line 1: nickname｜one-line identity (job/location)     ← index summary source
Body paragraphs, ordered by importance:
  - Interests / expertise
  - Communication style
  - Recent events (with @/# tags)
  - Interaction style with bot                         ← last
```

#### Group Memo Structure (~500 chars)

```
Line 1: group name｜member count｜one-line vibe        ← index summary source
Body paragraphs, ordered by stability:
  - Common topics / cultural norms (most stable)
  - Member relationships and roles (with @ tags)
  - Recent events (will decay, with @ participant tags)
```

#### Writing Rules (injected into compact prompt and tool descriptions)

1. Record **impressions and conclusions**, not logs
2. Tag people with `@QQ号`, groups with `#group_id` — **QQ号 is the sole identifier; nicknames are untrusted and may be spoofed**
3. First mention includes nickname `@123456(小明)`, never write a nickname without its QQ号
4. Personal facts → user memo; group event details → group memo
5. User memo ≤ 300 chars, group memo ≤ 500 chars
6. Only record self-disclosed or clearly observed facts — no speculation
7. Do not record what the bot itself said
8. If a nickname conflicts with a known user's name, **always verify by QQ号** before associating information

### Changelog (.log)

Append-only, one line per write. Never read during normal operation.

```
2026-03-31 14:30 | compact:group:987654 | 首次创建
2026-04-01 09:00 | compact:group:987654 | +学Rust信息
2026-04-02 15:30 | tool:session:a3f | 换工作到字节
```

### Write Safety

1. Write to `.md.tmp`
2. `rename(.tmp → .md)` — atomic on POSIX
3. Append one line to `.log` — fire-and-forget
4. Update in-memory cache

## 2. In-Memory Layer

### Data Model

```python
@dataclass
class Memo:
    id: str                          # "user_123456" / "group_987654"
    kind: Literal["user", "group"]
    identity: str                    # First content line (index summary source)
    body: str                        # Full prose body
    refs: set[str]                   # @/# references found in body
    updated: datetime
    source: str

class MemoStore:
    _memos: dict[str, Memo]             # Parsed structured objects
    _mentions: dict[str, set[str]]      # user_id → set of memo IDs mentioning them
    _lock_mgr: LockManager              # Per-file locks
```

### Parsing

Predictable structure — no complex parser needed:
- Line 1: HTML comment → `updated`, `source`
- First non-empty content line → `identity`
- Regex `@(\d+)` and `#(\d+)` → `refs`
- Remainder → `body`

### LockManager

Centralized lock allocation. All writers (compact, tool, Dream) use the same interface.

```python
class LockManager:
    _locks: defaultdict[str, asyncio.Lock]

    def get(self, id: str) -> asyncio.Lock:
        return self._locks[id]
```

Same-file writes serialize. Different-file writes run in parallel. No special privileges for Dream.

### Read API

| Method | Purpose | Complexity |
|--------|---------|-----------|
| `read(id) → Memo \| None` | Single memo | O(1) |
| `about(user_id) → list[Memo]` | All memos mentioning a user | O(1) via `_mentions` |
| `list_ids(kind?) → list[str]` | All user/group IDs | O(1) |
| `serialize_index() → str` | Render index.md content from `_memos` | O(n) |

### Write API

```python
async def write(self, id: str, memo: str, source: str) -> None:
    async with self._lock_mgr.get(id):
        # 1. .tmp → atomic rename
        # 2. append .log
        # 3. update _memos (re-parse)
        # 4. rebuild _mentions
        # 5. rewrite index.md
```

### Startup

```python
async def startup(self) -> None:
    # 1. Clean residual .tmp files
    # 2. Scan users/*.md + groups/*.md → parse into _memos
    # 3. Extract @/# from all memos → build _mentions
    # 4. Rebuild index.md from _memos (or validate existing)
```

## 3. Write Paths

### 3.1 Compact Extraction (automatic, both private and group)

Both `_compact()` (private) and `_compact_group()` (group) extract memos. Key change from current: **private compact also extracts memories** (currently only group does).

Compact prompt includes current memos as merge context and requests full rewrite (not append):

```
同时请更新相关用户的备忘录。当前备忘录：
  @123456: 「{current_memo_or_暂无记录}」
  @789012: 「{current_memo_or_暂无记录}」

{if group} 当前群备忘录：
  「{current_group_memo_or_暂无记录}」

输出 JSON：
{"summary": "...", "memos": {"123456": "...", ...}, "group_memo": "..."}
```

On compact, **batch all cache-breaking writes together** — update summary, memos, and group memo in the same operation. Do not let them split across multiple chat() calls.

#### Micro Compact (lightweight, no cache break)

When approaching token threshold but not critical:

```python
def _micro_compact(self, group_id):
    """Drop oldest 25% of messages. No LLM call, no summary change."""
    messages = self._timeline.get_messages(group_id)
    self._timeline.drop_oldest(group_id, len(messages) // 4)
```

Two thresholds:
- `token ratio 0.6` → micro compact (drop messages, preserve cache)
- `token ratio 0.8` → full compact (LLM summary, one-round cache miss)

#### Circuit Breaker

After 3 consecutive compact failures, stop retrying. Reset on next successful compact.

### 3.2 LLM Tool Calls (explicit, async)

**`recall_memo`** — flexible retrieval:

```python
recall_memo(
    id: str | None,            # Exact lookup: "user_123456"
    query: str | None,         # Fuzzy search by nickname/keyword
    kind: Literal["user", "group"] | None,
)
```

- `id` match → return full memo text
- `query` search → match against `identity` field first, then `body` full text. Return list of matching (id, identity) summaries. **Results always include the QQ号 so the LLM can disambiguate users with similar nicknames.** LLM picks interesting ones and calls again with `id` for full text

**`update_memo`** — async fire-and-forget:

```python
update_memo(
    id: str,                   # Target memo
    memo: str,                 # Full rewrite content
)
```

Executes via `asyncio.create_task`. Returns "已提交更新" immediately. Does not block the conversation. Source auto-tagged as `tool:{session_id}`.

### 3.3 Multi-Turn Memory Retrieval

The existing tool loop (`MAX_TOOL_ROUNDS=5`) supports multi-step memory lookups:

```
LLM sees index in system prompt → knows everyone at a glance
  → needs detail on @789012 → recall_memo(id="user_789012")
  → reads result, sees connection to @123456
  → recall_memo(id="user_123456") for more context
  → responds with full context
```

Tool calls within a round can execute in parallel via `asyncio.gather`.

## 4. Dream Agent

### Trigger

`N hours + M compacts since last Dream`, both conditions met. Values from config.

### Execution

Background `asyncio.create_task`. Independent LLM session with tools. Only one Dream instance at a time (`_dream_running` flag).

### System Prompt

```
你是记忆整理助手。以下是当前索引：
{index.md}

预检发现以下问题：
{programmatic pre-check results: dangling @refs, oversized memos, etc.}

请自主检查备忘录质量并修复问题。

Tools: recall_memo, update_memo
限制：最多 {max_rounds} 轮工具调用。
```

Pre-check is programmatic (scan refs, check sizes). Results injected as hints so the Dream agent has direction rather than scanning blindly.

### What Dream Does (LLM decides autonomously)

- Read memos with invalid @/# references → clean up
- Read oversized memos → consolidate and shorten
- Read related memos → fix cross-memo contradictions
- Verify index consistency → trigger rebuild if needed

All writes go through `MemoStore.write()` with `source="dream"`, using the same `LockManager`.

## 5. Prompt Caching Strategy

### Cache Breakpoint Layout (4 breakpoints)

```
Breakpoint ①: tools[-1]                                    ← Global shared, never changes
Breakpoint ②: system block 1 (personality+instruction+proactive) ← Global shared, never changes
Breakpoint ③: system block 2 (index + entity memo + context)     ← Per-entity
Breakpoint ④: messages[near-end]                                 ← Per-conversation
```

### PromptBuilder

```python
class PromptBuilder:
    _static_block: dict[str, Any]   # Built once at startup

    def build_static(self, identity: Identity, bot_self_id: str) -> None:
        """Called once at startup."""
        text = identity.personality
        text += f"\n\n【你的QQ号是 {bot_self_id}...】"
        if self._instruction:
            text += "\n\n" + self._instruction
        if identity.proactive:
            text += "\n\n" + identity.proactive
        self._static_block = {
            "type": "text", "text": text,
            "cache_control": {"type": "ephemeral"},   # Breakpoint ②
        }

    async def build_blocks(self, user_id, group_id, memo_store):
        blocks = [self._static_block]                 # Block 1: shared

        # Block 2: per-entity
        text = f"【全局索引】\n{memo_store.serialize_index()}"
        if group_id:
            memo = memo_store.read(f"group_{group_id}") or ""
            text += f"\n\n【当前在群 #{group_id} 中对话】\n{memo}"
        else:
            memo = memo_store.read(f"user_{user_id}") or ""
            text += f"\n\n【当前私聊 @{user_id}】\n{memo}"
        blocks.append({
            "type": "text", "text": text,
            "cache_control": {"type": "ephemeral"},   # Breakpoint ③
        })

        return blocks
```

### Multi-Level Cache Behavior

```
L1 (exact match): Same entity within 5 min → full prefix hit
L2 (lookback):    New/cold entity → ③ miss → lookback to ② → personality+instruction cached
L3 (lookback):    ② also miss → lookback to ① → tools cached
```

### Cache-Break Vectors

| Vector | Breaks | Runtime frequency |
|--------|--------|-------------------|
| personality/instruction change | ② | Never (startup only) |
| index update | ③ | Low (compact/dream) |
| entity memo update | ③ | Low (compact/tool) |
| New entity conversation | ③ | Per first chat |
| New message | ④ | Every turn (lookback handles) |
| Compact (summary rewrite) | ④ | Infrequent |
| Tool definition change | ① | Never (code change only) |

### Async update_memo Does Not Break Current Cache

`system_blocks` are built once at the start of `chat()`. Background `update_memo` writes to `_memos` cache but does not affect the already-built blocks. Cache break deferred to next `chat()` call.

### Compact Batches All Cache Breaks

When compact fires, update summary + memos + group memo together. One round of cache miss, not spread across multiple calls.

## 6. Agent Loop Changes

### Parallel Tool Execution

When LLM returns multiple tool calls in one round, execute concurrently:

```python
results = await asyncio.gather(*[
    self._tools.call(tu.name, json.dumps(tu.input), ctx=tool_ctx)
    for tu in tool_uses
])
```

### Circuit Breaker for Compact

```python
_compact_failures: int = 0
MAX_CONSECUTIVE_COMPACT_FAILURES = 3

async def _compact(self, ...):
    try:
        ...
        self._compact_failures = 0
    except Exception:
        self._compact_failures += 1
        if self._compact_failures >= MAX_CONSECUTIVE_COMPACT_FAILURES:
            logger.error("compact circuit breaker tripped")
            return
```

## 7. Docker Persistence

`storage/memories/` lives under the existing `./storage:/app/storage` volume mount in `docker-compose.yml`. No additional volume configuration needed — memos, logs, and index survive container restarts.

## 8. Migration (none)

No migration. Old `.qmd` files are ignored. The new memo system starts fresh with an empty `storage/memories/` directory.

## 8. Configuration

```python
class MemoConfig(BaseModel):
    dir: str = "storage/memories"
    user_max_chars: int = 300
    group_max_chars: int = 500
    index_max_lines: int = 200
    history_enabled: bool = True          # .log files

class CompactConfig(BaseModel):
    micro_ratio: float = 0.6             # Micro compact threshold
    full_ratio: float = 0.8              # Full compact threshold
    max_failures: int = 3                # Circuit breaker

class DreamConfig(BaseModel):
    interval_hours: int = 24
    min_compacts: int = 5                # Min compacts since last dream
    max_rounds: int = 15                 # Tool call limit
```

## 9. Instruction Chapter (to add to instruction.md)

The memo writing rules from section 1 should be added as a dedicated chapter in `soul/instruction.md` so the LLM always has them in context. Title: `## 备忘录书写规范`.
