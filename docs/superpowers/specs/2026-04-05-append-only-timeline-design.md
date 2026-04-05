# Append-Only Timeline for Cache Stability

## Problem

The group timeline (`GroupTimeline`) stores raw messages and re-merges them into
Anthropic format on every API call via `to_anthropic_messages()`. This has two
issues:

1. **No structural guarantee of prefix stability.** The Anthropic prompt cache
   requires byte-identical content up to the cache breakpoint. Currently the
   prefix stability relies on convention (only calling `append`), not on the
   data structure itself. Any future code path that modifies or inserts a
   message before the breakpoint silently breaks cache hits — a 12.5× cost
   difference (cache read is 0.1× base, write is 1.25×).

2. **Redundant merge on every call.** `to_anthropic_messages()` re-runs the
   same deterministic merge over the entire history each time, even though only
   the tail has changed.

## Design

Replace the raw message list with two structures:

- **`_TurnLog`** — append-only list of finalized Anthropic messages (alternating
  `user`/`assistant`). Once appended, entries are never modified. This is what
  gets sent to the API.
- **`pending`** — mutable buffer of raw `TimelineMessage` items accumulating the
  current user turn.

A new **SQLite table** (`group_messages`) persists every raw message for compact,
analysis, and future replay.

### Data Structures

```python
class _TurnLog(Sequence[dict[str, Any]]):
    """Append-only store of finalized Anthropic messages."""
    _data: list[dict[str, Any]]

    def append(self, turn: dict[str, Any]) -> None: ...
    def compact_truncate(self, count: int) -> None: ...
    def reset(self) -> None: ...
    # Sequence protocol: __getitem__, __len__, __bool__
    # NOT exposed: __setitem__, __delitem__, insert, pop, clear


class _GroupState:
    turns: _TurnLog                 # finalized Anthropic messages
    turn_times: list[float]         # parallel array: created_at per turn
    pending: list[TimelineMessage]  # accumulating raw user messages
    summary: str
    last_input_tokens: int
    last_cached_msg_index: int      # index into turns (not raw messages)
```

### Lifecycle

```
Group message arrives
  → pending.append(raw_msg)
  → SQLite INSERT (group_messages)

API call starts (_build_group_messages)
  → returns: [summary prefix] + list(turns) + [merge(pending)]
  → pending is NOT consumed; merge is a temporary computation
  → breakpoint placed within turns range (len-2)

Bot replies successfully (add role="assistant")
  → flush: merge(pending) → turns.append(user_turn), turn_times.append(now)
  → turns.append(assistant_turn), turn_times.append(now)
  → pending cleared

pass_turn
  → no change to turns or pending
  → pending preserved for next trigger
```

**Key invariant:** `turns` only grows inside `add(role="assistant")`, always
appending exactly two entries (one user turn + one assistant turn). Entries are
alternating and final.

**Why cache is stable:** `turns` entries are the exact bytes from previous API
calls. `_build_group_messages` extends them directly into the messages list — no
re-merge, no recomputation. The temporary `merge(pending)` is always the last
element, after the breakpoint.

```
Turn N:
  build → [S_u, S_a, T0, T1, ..., Tn, temp_merge_pending]
                                   ↑ breakpoint (len-2)

Turn N+1:
  build → [S_u, S_a, T0, ..., Tn, Tn+1, Tn+2, temp_merge_pending']
                                   ↑ old breakpoint, lookback hit
  Tn+1, Tn+2 are stored from flush — stable.
  temp_merge_pending' is after breakpoint — irrelevant to cache.
```

### Compact

```
needs_compact? (last_input_tokens > max_tokens × ratio)
  │
  ▼
Query SQLite: WHERE group_id=? AND created_at <= turn_times[split-1]
  → build conversation_text with speaker attribution
  → extract seen_user_ids (QQ numbers from speaker field)
  │
  ▼
LLM compression + append_memo tool calls
  │
  ▼
turns.compact_truncate(split)     ← only truncation path
turn_times = turn_times[split:]   ← parallel array stays in sync
summary = new_summary
last_cached_msg_index = 0         ← cache restarts
```

`split` unit changes from raw message count to turn count. Since turns alternate
user/assistant, `split=2` removes one complete conversation round.

Circuit breaker (`drop_oldest`) uses the same `compact_truncate` path.

### SQLite Raw Message Table

```sql
CREATE TABLE group_messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id     TEXT    NOT NULL,
    role         TEXT    NOT NULL,     -- 'user' | 'assistant'
    speaker      TEXT,                 -- '昵称(QQ号)' or NULL
    content_text TEXT,                 -- plain text extracted from content
    content_json TEXT,                 -- full Content JSON serialization
    message_id   INTEGER,             -- QQ message_id
    created_at   REAL    NOT NULL      -- time.time()
);

CREATE INDEX idx_gm_group_time ON group_messages(group_id, created_at);
```

**Two content columns:**

- `content_text` — concatenated text from all `TextBlock` entries. Used by
  compact to build conversation text without JSON deserialization. Sticker
  references (`«表情包:xxx»`) appear here as text.
- `content_json` — full `Content` serialization preserving `image_ref` blocks
  with local cache paths (`storage/image_cache/...`, `storage/stickers/...`).
  Enables future message reconstruction with images and stickers.

**Write timing:** every `GroupTimeline.add()` call writes to SQLite via
`asyncio.create_task` (fire-and-forget), consistent with the existing
`usage_tracker` pattern. SQLite writes are fast enough that ordering is
preserved in practice; message arrival order is also recorded via `created_at`.

**Retention:** compact does not delete SQLite records. They accumulate as a
persistent log for analysis and replay.

### Public API Changes

| Current method | Change |
|---|---|
| `add(group_id, role, content, speaker, message_id)` | Semantic change: `role="user"` → pending + SQLite; `role="assistant"` → flush pending + append turns + SQLite. Signature unchanged. |
| `get_messages(group_id)` | **Removed.** Replaced by `get_turns()` and `get_pending()`. |
| `to_anthropic_messages(group_id)` | **Removed.** `_build_group_messages` uses `get_turns()` + merge pending directly. |
| **New** `get_turns(group_id) → Sequence[dict]` | Read-only view of `_TurnLog`. |
| **New** `get_pending(group_id) → list[TimelineMessage]` | Copy of pending buffer. |
| `compact(group_id, split, new_summary)` | `split` unit is now turn count. |
| `drop_oldest(group_id, count)` | `count` unit is now turn count. |
| `clear(group_id)` | Clears turns + pending (reconnect reload). |
| `needs_compact`, `get_summary`, `*_input_tokens`, `*_cached_msg_index` | Unchanged. |

### Caller Impact

| File | Change |
|---|---|
| `client.py` `_build_group_messages` | Use `get_turns()` + merge pending instead of `to_anthropic_messages()`. |
| `client.py` `_compact_group` | Query SQLite for raw messages (replaces `get_messages()`). Adjust `split` semantics. |
| `client.py` `chat()` reply path | `add(role="assistant")` auto-flushes. No additional changes. |
| `chat/__init__.py` group_listener | `add(role="user")` signature unchanged. No changes. |
| `history_loader.py` | `add()` signature unchanged. Alternating calls naturally build turns. |
| `sticker_tools.py` | `add(role="user")` signature unchanged. No changes. |

### _build_group_messages Revised

```python
def _build_group_messages(self, group_id: str) -> list[dict[str, Any]]:
    assert self._timeline is not None
    messages: list[dict[str, Any]] = []

    # 1. Summary prefix (stable)
    summary = self._timeline.get_summary(group_id)
    if summary:
        messages.append({
            "role": "user",
            "content": [_cached_text(f"«对话摘要»\n{summary}")],
        })
        messages.append({
            "role": "assistant",
            "content": "好的，我已了解之前的对话内容。",
        })

    # 2. Turns — finalized, byte-identical to previous API calls
    turns = self._timeline.get_turns(group_id)
    messages.extend(turns)

    # 3. Pending — temporary merge as tail user message
    pending = self._timeline.get_pending(group_id)
    if pending:
        messages.append({
            "role": "user",
            "content": _merge_user_contents(pending),
        })

    # 4. Cache breakpoint within turns range
    cached_idx = self._timeline.get_cached_msg_index(group_id)
    if 0 < cached_idx < len(messages):
        target = messages[cached_idx]
        content = target.get("content")
        if isinstance(content, str):
            messages[cached_idx] = {
                "role": target["role"],
                "content": [_cached_text(content)],
            }
        elif isinstance(content, list):
            content = [*content]
            content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
            messages[cached_idx] = {"role": target["role"], "content": content}

    if len(messages) >= 2:
        self._timeline.set_cached_msg_index(group_id, len(messages) - 2)

    return messages
```
