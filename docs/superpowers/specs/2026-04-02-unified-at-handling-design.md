# Unified @ Handling Design

> Date: 2026-04-02
> Status: Draft

## Goal

Treat @bot and non-@ messages equally: @ triggers an LLM call but does NOT
guarantee a reply. The bot decides autonomously whether to respond via
`pass_turn`, just like a real group member.

## Current Behavior

| Scenario | Handler | `allow_skip` | Bot must reply? |
|----------|---------|-------------|-----------------|
| Group @bot | `chat` (priority=10, `to_me()`) | `False` | Yes |
| Group non-@ | `group_listener` → scheduler | `True` | No (pass_turn) |
| Private msg | `chat` (priority=10, `to_me()`) | `False` | Yes |

The `chat` handler interrupts the scheduler (`interrupt()`), runs its own
`_llm.chat()`, then releases (`release()`). Two completely separate code paths.

## New Behavior

| Scenario | Handler | pass_turn | Bot must reply? |
|----------|---------|-----------|-----------------|
| Group @bot | `group_listener` → scheduler (`is_at=True`) | Always active | No |
| Group non-@ | `group_listener` → scheduler (`is_at=False`) | Always active | No |
| Private msg | `private_chat` (standalone handler) | Always active | No |

All group messages flow through a single handler and a unified scheduler.
The `allow_skip` parameter is removed entirely; `pass_turn` is always honored.

## Design

### 1. Plugin Layer (`plugins/chat/__init__.py`)

**Delete** the `chat` handler (the `on_message(rule=to_me(), priority=10, block=True)`
matcher and `handle_chat()` function).

**Modify** `group_listener` / `collect_group_context()`:
- Instead of returning early on `event.is_tome()`, call
  `_scheduler.notify(group_id, is_at=True)`.
- Non-@ messages continue to call `_scheduler.notify(group_id, is_at=False)`.

**Add** a new `private_chat` handler for private (DM) messages:
- `on_message(rule=to_me(), priority=10, block=True)` — but only handles
  non-`GroupMessageEvent` events.
- Calls `_llm.chat()` directly (no scheduler involvement).
- `pass_turn` is always active (bot can choose not to reply in DMs too).

### 2. Scheduler (`llm/scheduler.py`)

**Delete**: `interrupt()`, `release()` methods.

**Delete** from `_GroupSlot`: `interrupted` field.

**Add** to `_GroupSlot`: `pending_at: bool = False`.

**Modify** `notify(group_id, is_at=False)`:

```
notify(group_id, is_at):
    msg_count += 1

    if is_at:
        if running_task is active:
            pending_at = True       # queue for merge
            return
        else:
            cancel debounce
            _fire() immediately
            return

    # Non-@ path (unchanged logic)
    if running_task is active:
        return                      # skip, will catch up later
    if msg_count >= batch_size:
        _fire() immediately
    else:
        reset debounce timer
```

**Modify** `_do_chat()` finally block:

```python
finally:
    slot.running_task = None
    if slot.pending_at:
        slot.pending_at = False
        self._fire(group_id)        # immediately run merged @ call
```

**Effect on multiple @**:
- First @ fires immediately.
- Subsequent @ messages during execution set `pending_at = True`.
- When the first call finishes, a single merged call runs. The timeline
  already contains all queued @ messages, so they are naturally merged.

**Effect on @ vs debounce/batch**:
- Both share `running_task` as a mutual exclusion lock.
- No special priority — first come, first served.
- If a debounce fires while an @ call is running, it simply waits
  (the messages accumulate in the timeline).

### 3. LLM Client (`llm/client.py`)

**Delete** `allow_skip` parameter from `chat()` signature.

**Simplify** pass_turn logic:

```python
# Before:
if pass_turn and allow_skip:
    ...
    return None
if pass_turn:
    tool_uses = [tu for tu in tool_uses if tu.name != "pass_turn"]

# After:
if pass_turn:
    ...
    return None
```

The `_PASS_TURN_TOOL` definition, tool list construction, and all other
`chat()` logic remain unchanged.

### 4. Soul Instructions (`soul/identity.md`)

Update the proactive section. Key change:

**Before:**
> 只有以下情况才插话：
> - 有人 @ 你或明确在跟你说话

**After:**
> 只有以下情况才插话：
> - 有人明确在跟你说话或向你提问
>
> 有人 @ 你表示对方在跟你说话，但这不意味着你必须回复——同样按以上标准自行判断。

`soul/instruction.md` does not need changes — its existing "自由决定是否参与群聊"
wording is already consistent with this design.

## Files Changed

| File | Change |
|------|--------|
| `src/plugins/chat/__init__.py` | Delete `chat` handler; modify `group_listener` to pass `is_at` to scheduler; add `private_chat` handler |
| `src/llm/scheduler.py` | Delete `interrupt()`/`release()`/`interrupted`; add `pending_at`; modify `notify()` signature and `_do_chat()` finally block |
| `src/llm/client.py` | Delete `allow_skip` parameter; simplify pass_turn branch |
| `soul/identity.md` | Update proactive rules — @ is not a guaranteed reply |
| Tests | Update scheduler and client tests to match new signatures |

## Out of Scope

- Changing debounce/batch config values.
- Changing the `pass_turn` tool definition or prompt engineering beyond the
  specified identity.md change.
- Private message scheduling (DMs remain direct calls, no queue).
