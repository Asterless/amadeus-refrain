# Architecture Details

## Key Design Decisions

- **Raw Anthropic API via aiohttp SSE** — no SDK. `src/llm/client.py` manually parses SSE `data:` lines to extract text deltas and tool_use blocks. Adding tool calls means touching the `_call_api` function.
- **Prompt Caching strategy** — System blocks (identity + instruction, user memory) use `cache_control: ephemeral`. Group chat context goes into messages (not system) because it changes every turn. The second-to-last history message gets `cache_control` to maximize cache hits on conversation prefix. Tool definitions also cached (last tool gets `cache_control`).
- **Context window management** — When estimated input tokens exceed `max_context_tokens × compact_ratio`, the front half of history is compressed into a summary via a separate LLM call. Group compaction also extracts user traits/events into long-term memory.
- **Segmented responses** — Bot replies can contain `---` separators; each segment is sent as a separate QQ message with a short delay.
- **Tool framework** — Tools extend `src/tools/base.py:Tool` ABC (name, description, parameters as JSON Schema, async execute). Registered in `ToolRegistry`, converted to Anthropic format via OpenAI-style intermediate. `ToolContext` carries the Bot instance and event metadata.
- **Soul directory** — `soul/` holds personality & instruction configs. `identity.md` defines a single persona (Markdown: `# Name` heading for the persona name, body for personality, optional `## proactive` section for proactive chat rules). `instruction.md` holds behavioral directives injected into the system prompt.
- **Memory layers** — Short-term: in-memory deque per session (private chat). Long-term: `.qmd` files per user in `storage/memories/` with profile + events sections. Group timeline: in-memory deque of recent messages per group (`GroupTimeline`), with summary from compaction.
- **Session IDs** — `group_{group_id}` for group chats, `private_{user_id}` for DMs.
- **History bootstrap** — On bot connect, `history_loader.py` pulls recent messages from NapCat HTTP API for all groups, populating the group timeline. After loading, the scheduler fires once per group to catch up on missed messages.

## Proactive Chat (GroupChatScheduler)

The bot can autonomously join group conversations when the identity has a `proactive` field defined. The `GroupChatScheduler` (`src/llm/scheduler.py`) manages this:

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
| `llm.context` | `max_context_tokens`, `compact_ratio` | Context window & compaction threshold |
| `group` | `max_timeline_messages`, `history_load_count`, `allowed_groups`, `debounce_seconds`, `batch_size` | Group chat behavior & scheduler |
| `napcat` | `api_url` | NapCat HTTP API endpoint |
| `memory` | `dir` | Long-term memory storage path |
| `soul` | `dir` | Soul config directory |
| `log` | `dir` | Log directory |
| top-level | `superusers`, `allowed_private_users` | Access control |

NoneBot itself is configured in `pyproject.toml` under `[tool.nonebot]`.

## Available Tools

| Tool | Description |
|------|-------------|
| `DateTimeTool` | Current date/time |
| `WebFetchTool` | Fetch web page content |
| `HttpApiTool` | Call NapCat HTTP API |
| `SaveMemoryTool` | Save user memory |
| `RecallMemoryTool` | Recall user memory |
| `MuteUserTool` | Mute group member (superuser only) |
| `SetTitleTool` | Set member title (superuser only) |
| `SendGroupMsgTool` | Send group message (superuser only) |
| `pass_turn` | Skip this turn (injected by LLMClient for scheduler calls, not a registered tool) |

## Message Rendering

- **Reply quotes**: when a message replies to another, the text includes `[回复 昵称(QQ号): 原文摘要]` prefix
- **@mentions**: `[CQ:at,qq=123]` segments are rendered as `@123` in the text sent to the model
- **Bot self-ID**: injected into system prompt so the model knows which messages are its own

## Access Control

- `allowed_groups`: group whitelist (empty = allow all)
- `allowed_private_users`: private chat whitelist (empty = allow all)
- Superuser tools (Mute/SetTitle/SendGroupMsg) check `superusers` set from config
