# LLM Usage Tracking Design

## Goal

Record every LLM API call's token consumption and latency to a local SQLite database. Provide CLI and HTTP query interfaces for usage summaries. Alert on anomalies via loguru WARNING + Bot private message to admins.

## Non-goals

- Cost/budget alerting (may add later)
- Prometheus/OpenTelemetry integration
- Multimodal token breakdown (API doesn't provide it)

---

## 1. Data Model

Single table `llm_calls` in `storage/usage.db`:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ts | TEXT | ISO8601 timestamp |
| call_type | TEXT | `chat` / `proactive` / `compact` / `dream` |
| user_id | TEXT | QQ number (nullable) |
| group_id | TEXT | Group number (nullable) |
| model | TEXT | Model name |
| input_tokens | INTEGER | Non-cached input tokens (raw API value) |
| cache_read_tokens | INTEGER | Tokens read from prompt cache |
| cache_create_tokens | INTEGER | Tokens written to prompt cache |
| output_tokens | INTEGER | Output tokens |
| tool_rounds | INTEGER | Number of tool loop iterations |
| elapsed_s | REAL | Wall-clock seconds |
| error | TEXT | Error message (nullable) |

Indexes: `ts`, `user_id`, `group_id`, `call_type`.

---

## 2. Architecture

### 2.1 New module: `src/llm/usage.py`

**`UsageTracker`** — singleton, initialized at bot startup.

```
_call_api() returns
       │
       ▼
  LLMClient collects per-chat totals (across tool rounds)
       │
       ▼
  UsageTracker.record()  ──→  SQLite write
       │
       ├─ check cache hit rate → WARNING + Bot PM if low
       ├─ check elapsed_s → WARNING + Bot PM if slow
       └─ check error → WARNING + Bot PM
```

Key decisions:

- **`record()` is async, fire-and-forget** — called via `asyncio.create_task()` from `LLMClient.chat()` after the reply is sent. Write failure does not affect the user.
- **Token accumulation across tool rounds** — `LLMClient.chat()` sums `input_tokens`, `output_tokens`, `cache_read`, `cache_create` across all `_call_api()` calls in a single `chat()` invocation, then records once.
- **Bot PM for alerts** — `UsageTracker` accepts an optional async callback `alert_fn(msg: str)` set after Bot connects. The plugin wires this to `Bot.send_private_msg()` targeting admin QQ numbers from `BotConfig.admins`.

### 2.2 Fix: extract `output_tokens` from SSE stream

Current `_call_api()` only reads usage from `message_start`. The real `output_tokens` arrives in `message_delta` events. Fix: merge usage from `message_delta` into the usage dict (later values overwrite).

After fix, `_call_api()` return dict adds:

```python
"output_tokens": usage.get("output_tokens", 0),
```

### 2.3 Consolidate cache hit rate checking

Current `LLMClient._check_cache_rate()` (lines 245-277) moves into `UsageTracker.record()`. The method and `_DEFAULT_CACHE_HIT_WARN` constant are deleted from `client.py`. The `cache_hit_warn` config value is passed to `UsageTracker` instead.

The existing loguru cache stats line in `_call_api()` (lines 172-176) is also removed — `UsageTracker.record()` logs all usage info in one unified line.

---

## 3. Alerting

Alerts fire inside `UsageTracker.record()`. Each alert emits a loguru WARNING **and** calls `alert_fn` to PM admins.

| Alert | Condition | Throttle |
|-------|-----------|----------|
| Cache hit rate low | `cache_read / total_input * 100 < cache_hit_warn` | Per call (same as current behavior) |
| Slow call | `elapsed_s > slow_threshold_s` | Per call |
| API error | `error is not None` | Per call |

No daily/hourly throttling for now — these are per-call checks on anomalous conditions that should be rare.

---

## 4. Configuration

New `UsageConfig` in `src/config.py`:

```toml
[llm.usage]
enabled = true
slow_threshold_s = 60.0
```

`cache_hit_warn` stays in `[compact]` since it's conceptually a compact/cache concern, but `UsageTracker` reads it from there.

`BotConfig.admins` (existing) is used for alert targets — no new config needed.

---

## 5. Query Interfaces

### 5.1 CLI: `src/llm/usage_cli.py`

Runnable as `uv run python -m src.llm.usage_cli`. Subcommands:

- `today` — today's totals (calls, tokens, by type)
- `month [YYYY-MM]` — monthly summary
- `top-users [--days N]` — top users by token consumption
- `top-groups [--days N]` — top groups by token consumption

Output: formatted table to stdout via plain `print()`. No new dependency.

### 5.2 HTTP: `/api/usage` endpoint

NoneBot uses FastAPI under the hood. Add a router in `src/llm/usage_routes.py`:

- `GET /api/usage/today` — JSON: today's totals
- `GET /api/usage/month?month=YYYY-MM` — JSON: monthly summary
- `GET /api/usage/top-users?days=N` — JSON: top users
- `GET /api/usage/top-groups?days=N` — JSON: top groups

Registered at startup in the plugin `__init__.py` via `nonebot.get_app()`.

---

## 6. Call Sites

Where `UsageTracker.record()` is called and with what `call_type`:

| Call site | call_type | Notes |
|-----------|-----------|-------|
| `LLMClient.chat()` after reply sent | `chat` or `proactive` | Distinguish by `allow_skip` param: `True` = proactive |
| `LLMClient._compact()` | `compact` | After compact API call |
| `LLMClient._compact_group()` | `compact` | After compact API call |
| `DreamAgent` | `dream` | When dream consolidation runs |

For `chat()`: token counts are **summed across all tool rounds** in a single `chat()` invocation. One record per `chat()` call, not per `_call_api()` call. This gives the true cost of answering one user message.

For `compact` calls: each `_call_api()` inside `_compact()` / `_compact_group()` records separately since they are independent LLM calls.

---

## 7. Dependencies

- **New**: `aiosqlite` — async SQLite driver
- **No other new dependencies**

---

## 8. File Changes Summary

| File | Change |
|------|--------|
| `src/llm/usage.py` | **New** — `UsageTracker` class |
| `src/llm/usage_cli.py` | **New** — CLI query tool |
| `src/llm/usage_routes.py` | **New** — FastAPI routes |
| `src/llm/client.py` | Fix `output_tokens` extraction from `message_delta`; remove cache log + `_check_cache_rate()`; add `UsageTracker` integration; accumulate tokens across tool rounds |
| `src/config.py` | Add `UsageConfig` |
| `src/plugins/chat/__init__.py` | Initialize `UsageTracker`; wire alert callback on bot connect |
| `pyproject.toml` | Add `aiosqlite` dependency |
| `config.example.toml` | Add `[llm.usage]` section |
