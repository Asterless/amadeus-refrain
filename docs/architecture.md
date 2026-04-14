# Architecture Details

## Key Design Decisions

- **Raw Anthropic API via aiohttp SSE** — no SDK. `src/llm/client.py` manually parses SSE `data:` lines to extract text deltas and tool_use blocks. Adding tool calls means touching the `_call_api` function.
- **Prompt Caching strategy** — 4 cache breakpoints: ① tools[-1] (global shared), ② system block 1: personality + instruction + admins + proactive rules (global shared, built once at startup), ③ system block 2: memo index + entity memo + sticker library (per-entity), ④ messages[near-end] (per-conversation). Group timeline summaries inserted before messages for cache stability. Tool definitions also cached (last tool gets `cache_control`).
- **Context window management** — When estimated input tokens exceed `max_context_tokens × compact_ratio`, the front half of history is compressed into a summary via a separate LLM call. During compaction, the LLM receives an `append_memo` tool to extract user traits/events into long-term memory (§ Compact Memo Extraction). A circuit breaker drops oldest messages after `max_failures` consecutive compact failures.
- **Segmented responses** — Bot replies can contain `---cut---` separators; each segment is sent as a separate QQ message with a 0.5s delay.
- **Tool framework** — Tools extend `src/tools/base.py:Tool` ABC (name, description, parameters as JSON Schema, async execute). Registered in `ToolRegistry`, converted to Anthropic format via OpenAI-style intermediate. `ToolContext` carries the Bot instance and event metadata. Tools are executed in parallel within each round.
- **Soul directory** — `soul/` holds personality & instruction configs. `identity.md` defines a single persona (Markdown: `# Name` heading for the persona name, body for personality, optional `## 插话方式` section for proactive chat rules — exact heading match required). `instruction.md` holds behavioral directives injected into the system prompt.
- **Memory layers** — Short-term: in-memory deque per session (private chat). Long-term: `.md` files per user/group in `storage/memories/` with profile + events sections + `## 待整理` pending section (auto-filled by compact memo extraction, organized by Dream agent). Group timeline: append-only turns + pending buffer per group (`GroupTimeline`), with summary from compaction and SQLite persistence via `MessageLog`. Max 200 groups in memory (LRU eviction).
- **Session IDs** — `group_{group_id}` for group chats, `private_{user_id}` for DMs.
- **History bootstrap** — On bot connect, `history_loader.py` pulls recent messages from NapCat HTTP API for all groups, populating the group timeline (with image caching and sticker recognition). After loading, the scheduler fires once per group to catch up on missed messages.

## Proactive Chat (GroupChatScheduler)

The bot can autonomously join group conversations when the identity has a `## 插话方式` section defined. The `GroupChatScheduler` (`src/llm/scheduler.py`) manages this:

- **Debounce**: after each non-@ group message, a debounce timer starts (`debounce_seconds`). If the group goes quiet, the scheduler triggers an LLM call.
- **Batch**: if messages accumulate to `batch_size` before the debounce fires, the scheduler triggers immediately.
- **@bot interrupt**: when someone @s the bot, the scheduler cancels any pending debounce/running proactive task for that group, yielding to the direct @bot handler. After the @bot reply completes, the scheduler is re-enabled.
- **pass_turn tool**: when the scheduler fires, the LLM receives the `pass_turn` tool. If the model decides there's nothing worth saying, it calls `pass_turn` and no message is sent.
- **Startup catch-up**: on bot connect, the scheduler triggers once for each group that has history, so the bot can respond to messages it missed while offline.

## Config

Config flows through `src/config.py:BotConfig` (Pydantic model), loaded via `src/config_loader.py`. Priority (low → high):

1. Pydantic defaults
2. TOML file (`config.toml` or `BOT_CONFIG_PATH` env var)
3. Environment variables (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `NAPCAT_API_URL`)
4. CLI arguments (via `bot.py` argparse)

Key config sections:

| Section | Fields | Purpose |
|---------|--------|---------|
| `llm` | `base_url`, `api_key`, `model`, `max_tokens` | LLM API connection |
| `llm.context` | `max_context_tokens` | Context window size |
| `llm.usage` | `enabled`, `slow_threshold_s` | Usage tracking & slow call alerts |
| `compact` | `ratio`, `compress_ratio`, `max_failures`, `cache_hit_warn`, `cache_alert_window_m`, `cache_alert_cooldown_m` | Context compaction & cache alerting |
| `dream` | `enabled`, `interval_hours`, `max_rounds` | Dream agent (periodic memo consolidation) |
| `group` | `history_load_count`, `allowed_groups`, `debounce_seconds`, `batch_size`, `at_only`, `blocked_users`, `overrides` | Group chat behavior, scheduler & per-group overrides |
| `napcat` | `api_url` | NapCat HTTP API endpoint |
| `memo` | `dir`, `user_max_chars`, `group_max_chars`, `index_max_lines`, `history_enabled` | Long-term memo storage |
| `soul` | `dir` | Soul config directory |
| `log` | `dir` | Log directory |
| `vision` | `enabled`, `max_images_per_message`, `max_dimension`, `cache_dir`, `cache_max_age_hours` | Multimodal image understanding |
| `sticker` | `enabled`, `storage_dir`, `max_count` | Sticker library |
| top-level | `admins`, `allowed_private_users` | Access control & admin designation |

`admins` is a `dict[str, str]` mapping QQ numbers to nicknames. Admins are injected into the system prompt as trusted sources and authorized for group admin tools.

NoneBot itself is configured in `pyproject.toml` under `[tool.nonebot]`.

### Per-Group Config

`group.overrides` maps group IDs to `GroupOverride`, allowing per-group tuning of `at_only`, `debounce_seconds`, `batch_size`, `history_load_count`, and `blocked_users`. Resolved via `GroupConfig.resolve(group_id) -> ResolvedGroupConfig`:

- `blocked_users`: union of global + per-group lists (additive, not override)
- All other fields: per-group value if set, else global default

### Message Log

`MessageLog` (`src/memory/message_log.py`) persists every raw group message to SQLite (`storage/messages.db`). Writes are fire-and-forget via `asyncio.create_task`. The `group_messages` table stores `group_id`, `role`, `speaker`, `content_text`, `content_json`, `message_id`, `created_at` with an index on `(group_id, created_at)`. Used by `_compact_group` to query raw messages with speaker info for LLM compression.

## Available Tools

| Tool | Class | Description |
|------|-------|-------------|
| `recall_memo` | `RecallMemoTool` | Recall user/group memo by exact id or fuzzy query |
| `update_memo` | `UpdateMemoTool` | Overwrite user/group memo (async fire-and-forget) |
| `get_datetime` | `DateTimeTool` | Current date/time (Asia/Shanghai) |
| `web_fetch` | `WebFetchTool` | Fetch web page content (SSRF-protected) |
| `web_search` | `WebSearchTool` | DuckDuckGo web search (max 10 results) |
| `http_api` | `HttpApiTool` | Call NapCat HTTP API |
| `mute_user` | `MuteUserTool` | Mute group member (admin only; duration=0 unmutes) |
| `set_title` | `SetTitleTool` | Set member special title (admin only) |
| `send_group_msg` | `SendGroupMsgTool` | Send group message (admin only) |
| `save_sticker` | `SaveStickerTool` | Save image to sticker library (conditional on sticker enabled) |
| `manage_sticker` | `ManageStickerTool` | Update description/usage_hint or delete sticker (delete is admin only; conditional on sticker enabled) |
| `send_sticker` | `SendStickerTool` | Send sticker as image message (conditional on sticker enabled) |
| `pass_turn` | — | Skip this turn (injected by LLMClient for all chat calls, not a registered tool) |
| `append_memo` | — | Append observation to memo pending section (injected only during compaction, not a registered tool) |
| `list_stickers` / `delete_sticker` | — | Dream-only tools defined inline in `dream.py` for sticker library curation |

## Group Timeline

Append-only conversation timeline per group (`src/memory/group_timeline.py`).

- **`_TurnLog`**: immutable `Sequence` of finalized Anthropic messages. Supports append and truncation (for compaction), but not arbitrary mutation.
- **`pending`**: mutable buffer of raw `TimelineMessage` dicts accumulating the current user turn. Flushed into `_TurnLog` when an assistant reply arrives.
- **`_GroupState`**: per-group state holding `turns`, `pending`, `turn_times`, `summary`, `last_input_tokens`, `last_cached_msg_index`.
- **Cache stability**: the turns range is byte-identical between calls; the prompt cache breakpoint is placed at `len(messages) - 2`, so only the newest pending merge invalidates the cache.
- **SQLite backing**: every `add()` call also records the raw message via `MessageLog.record()` (fire-and-forget), enabling `_compact_group` to query historical messages with speaker info.
- **Compaction**: `compact(split, new_summary)` truncates turns at a split point (turn count) and stores a new summary. `drop_oldest(count)` is the circuit-breaker fallback.
- **LRU eviction**: max 200 groups in memory; least-recently-used groups evicted on overflow.

## Vision System

Multimodal image understanding, enabled by default (`vision.enabled = true`).

**Pipeline**: QQ image segment → download URL concurrently → downscale via pyvips to `max_dimension` (768px default) → cache to disk as JPEG → send as base64 in Anthropic `image` content blocks.

- **Image cache** (`src/memory/image_cache.py`): two-level hash directory layout (`ab/abc123def456.jpg`), 8 concurrent downloads max, auto-cleanup on startup for images older than `cache_max_age_hours`.
- **Per-message limit**: `max_images_per_message` (default 5), excess images rendered as `[图片]` text.
- **Fallback**: when vision is disabled or images fail to download, images are rendered as `[图片]` or `[summary]` text.
- **Sticker recognition**: during history loading, downloaded images are checked against the sticker library by content hash; matches use the sticker path instead of the image cache.

## Sticker System

Persistent sticker library for the bot to collect and send image stickers.

- **Storage** (`src/sticker/store.py`): images stored in `storage/stickers/` with `index.json` metadata. Content-hash dedup via SHA256 prefix (`stk_{hash}`). Supports JPG, PNG, WebP; rejects GIF.
- **Tools**: `SaveStickerTool` (requires image_ref, description, usage_hint) and `SendStickerTool` (sends by sticker_id). Both conditional on `sticker.enabled`.
- **Prompt integration**: sticker library summary injected into system block 2 via `StickerStore.format_prompt_view()`.
- **Dream curation**: Dream agent can list and delete stickers, pruning low-usage or inaccurately described entries. Max count enforced via `sticker.max_count`.

## Dream Agent

Background agent for periodic memory consolidation (`src/llm/dream.py`).

- **Schedule**: runs on `dream.interval_hours` interval (default 24h, first run after one full interval). Disabled by default (`dream.enabled = false`).
- **Tasks**: (1) merge pending items from `## 待整理` into structured memo sections, (2) cross-file validation of references, (3) fix structural issues (dangling refs, oversized memos), (4) sticker library curation (if enabled).
- **Pre-check**: `dream_pre_check()` programmatically scans for structural issues before the LLM loop.
- **Tool loop**: up to `max_rounds` (default 15) rounds with `recall_memo`, `update_memo`, `list_stickers`, `delete_sticker` tools.
- **Logging**: dedicated log sink (`storage/logs/dream_*.log`), filtered from main bot log.
- **Lifecycle**: started on bot connect, stopped on shutdown.

## Usage Tracking

SQLite-backed recording of all LLM API calls (`src/llm/usage.py`).

- **Database**: `storage/usage.db` with `llm_calls` table. Records: timestamp, call_type (`chat`/`proactive`/`compact`/`dream`), user_id, group_id, model, input/output/cache_read/cache_create tokens, tool_rounds, elapsed_s, error.
- **Alerting**: PMs admin users when (1) average cache hit rate drops below `cache_hit_warn` % over `cache_alert_window_m` minutes, or (2) a single call exceeds `slow_threshold_s` seconds. Alert cooldown prevents spam.
- **API** (`src/llm/usage_routes.py`): FastAPI routes mounted on the NoneBot app: `/usage/summary/today`, `/usage/summary/month`, `/usage/top-users`, `/usage/top-groups`, `/usage/timeseries`.
- **TUI** (`src/llm/usage_tui.py`): Rich-based interactive dashboard via `uv run python -m src.llm.usage_cli tui day|week|month [date]`.

## Compact Memo Extraction

During context compaction, the LLM receives an `append_memo` tool that allows it to extract new observations into long-term memory:

- **Private chat**: extracts user traits/preferences into `user_{user_id}` memo. Source tagged as `compact:private:{session_id}`.
- **Group chat**: extracts user traits and group dynamics into both `user_{uid}` and `group_{group_id}` memos. Collects all user IDs seen in compacted messages. Source tagged as `compact:group:{group_id}`.
- **Pending section**: observations are appended to the `## 待整理` area of the target memo, later organized by the Dream agent.
- **Circuit breaker**: after `max_failures` (default 3) consecutive compact failures, compaction falls back to dropping oldest messages instead of calling the LLM.

## Message Rendering

- **Reply quotes**: when a message replies to another, the text includes `[回复 昵称(QQ号): 原文摘要]` prefix (50 char cap, 200 for bot's own messages)
- **@mentions**: `[CQ:at,qq=123]` segments are rendered as `@123` in the text sent to the model; self-@ rendered as `@我`
- **Face emoji**: `[CQ:face,id=X]` converted to text representation via `face_to_text()` from `src/constants/qq_face.py`
- **Images**: downloaded concurrently, cached, and sent as `image_ref` content blocks (resolved to base64 before API call); excess images or failures rendered as `[图片]`
- **Bot self-ID**: injected into system prompt so the model knows which messages are its own
- **CQ code normalization**: malformed CQ codes (`[CQ:reply,id:123]`) auto-fixed to `[CQ:reply,id=123]`

## Access Control

- `allowed_groups`: group whitelist (empty = allow all)
- `allowed_private_users`: private chat whitelist (empty = allow all)
- `admins`: QQ→nickname dict injected into system prompt as trusted sources; authorized for admin tools (MuteUser, SetTitle, SendGroupMsg, SaveSticker with `admin` source tag)
