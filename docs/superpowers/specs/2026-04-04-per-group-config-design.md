# Per-Group Configuration

## Problem

All group chat settings are global. There is no way to:

- Block a specific user in one group
- Make the bot @-only in one group while proactive in another
- Tune debounce/batch timing per group

## Design

### Config model

Add `GroupOverride` for per-group settings and a `ResolvedGroupConfig` for the merged result.

```python
class GroupOverride(BaseModel):
    blocked_users: list[int] = []
    at_only: bool | None = None
    debounce_seconds: float | None = None
    batch_size: int | None = None
    history_load_count: int | None = None

class GroupConfig(BaseModel):
    history_load_count: int = 30
    allowed_groups: list[int] = []
    debounce_seconds: float = 5.0
    batch_size: int = 10
    at_only: bool = False
    blocked_users: list[int] = []
    overrides: dict[int, GroupOverride] = {}

class ResolvedGroupConfig(BaseModel):
    blocked_users: set[int]       # global UNION per-group
    at_only: bool
    debounce_seconds: float
    batch_size: int
    history_load_count: int
```

`GroupConfig.resolve(group_id: int) -> ResolvedGroupConfig`:

- Scalar fields: per-group value if not `None`, else global value
- `blocked_users`: union of global and per-group lists

### TOML format

```toml
[group]
history_load_count = 30
allowed_groups = []
debounce_seconds = 5.0
batch_size = 10
at_only = false
blocked_users = []

[group.overrides.100001]
blocked_users = [123456, 789012]
at_only = true

[group.overrides.100002]
debounce_seconds = 10.0
batch_size = 20
history_load_count = 50
```

### Message pipeline changes

In `collect_group_context`, after the `allowed_groups` check:

```
resolved = group_config.resolve(event.group_id)
if event.user_id in resolved.blocked_users:
    return  # invisible to bot: no timeline, no scheduling
```

Blocked user messages are completely invisible — not recorded, not counted, not triggering.

### Scheduler changes

`GroupChatScheduler.notify()` uses resolved config per group:

- **`at_only = True`**: non-@ messages still enter the timeline (added before `notify`), increment `msg_count`, but do NOT start debounce or batch triggers. Only `is_at=True` fires the LLM call.
- **`debounce_seconds` / `batch_size`**: use the resolved per-group values instead of the global constructor values.

The scheduler either holds a reference to `GroupConfig` to resolve on each call, or receives the resolved config as a parameter from the listener.

### History loader changes

`_on_connect` resolves `history_load_count` per group before calling `load_group_history`. The loader needs to accept per-group counts (e.g., a `dict[str, int]` mapping group_id to count) instead of a single global count.

### Unchanged

- `allowed_groups` semantics unchanged; takes priority over overrides
- `GroupTimeline`, `LLMClient`, `DreamAgent` unaffected
- Private chat flow unaffected

## Files to modify

| File | Change |
|------|--------|
| `src/config.py` | Add `GroupOverride`, `ResolvedGroupConfig`, new fields on `GroupConfig`, `resolve()` method |
| `config.example.toml` | Add new global fields and example `[group.overrides.*]` sections |
| `src/plugins/chat/__init__.py` | Pass `GroupConfig` to scheduler; add `blocked_users` filter in `collect_group_context`; per-group `history_load_count` in `_on_connect` |
| `src/llm/scheduler.py` | Accept `GroupConfig` ref; resolve per-group `at_only`, `debounce_seconds`, `batch_size` in `notify()` |
| `src/memory/history_loader.py` | Accept per-group count instead of single global count |
| `tests/` | Tests for `resolve()` logic, blocked user filtering, at_only scheduler behavior |
