# Architecture Details

## Key Design Decisions

- **Raw Anthropic API via aiohttp SSE** — no SDK. `src/llm/client.py` manually parses SSE `data:` lines to extract text deltas and tool_use blocks. Adding tool calls means touching the `_call_api` function.
- **Prompt Caching strategy** — System blocks (identity + instruction, user memory) use `cache_control: ephemeral`. Group chat context goes into messages (not system) because it changes every turn. The second-to-last history message gets `cache_control` to maximize cache hits on conversation prefix. Tool definitions also cached (last tool gets `cache_control`).
- **Cache warming** — `LLMClient.maybe_warm()` sends a background `max_tokens=1` API call after every N group messages to keep the prompt cache hot. Configurable via `warm_enabled`, `warm_interval_messages`, `warm_ttl_seconds`.
- **Context window management** — When estimated input tokens exceed `max_context_tokens × compact_ratio`, the front half of history is compressed into a summary via a separate LLM call. Group compaction also extracts user traits/events into long-term memory.
- **Segmented responses** — Bot replies can contain `---` separators; each segment is sent as a separate QQ message with a short delay.
- **Tool framework** — Tools extend `src/tools/base.py:Tool` ABC (name, description, parameters as JSON Schema, async execute). Registered in `ToolRegistry`, converted to Anthropic format via OpenAI-style intermediate. `ToolContext` carries the Bot instance and event metadata.
- **Soul directory** — `soul/` holds personality & instruction configs. `identities.md` defines personas (Markdown with `## id` sections + `- key: value` metadata); `instruction.md` holds behavioral directives injected into the system prompt. Resolution order: manual override > keyword/group match (by priority) > default.
- **Memory layers** — Short-term: in-memory deque per session. Long-term: `.qmd` files per user in `storage/memories/` with profile + events sections. Group timeline: in-memory deque of recent messages per group (`GroupTimeline`).
- **Session IDs** — `group_{group_id}` for group chats, `private_{user_id}` for DMs. Used as keys for short-term memory and identity overrides.
- **History bootstrap** — On bot connect, `history_loader.py` pulls recent messages from NapCat HTTP API for all groups, populating both short-term memory and group timeline.

## Config

Config flows through `src/config.py:BotConfig` (Pydantic model), loaded via `src/config_loader.py`. Priority (low → high):

1. Pydantic defaults
2. TOML file (`config.toml` or `BOT_CONFIG_PATH` env var)
3. Environment variables (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `NAPCAT_API_URL`)
4. CLI arguments (via `bot.py` argparse)

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

## User Commands

- `/identity list` — list available personas
- `/identity <id>` — switch to a specific persona
- `/identity reset` — restore auto-matching

## Access Control

- `allowed_groups`: group whitelist (empty = allow all)
- `allowed_private_users`: private chat whitelist (empty = allow all)
- Superuser tools (Mute/SetTitle/SendGroupMsg) check `superusers` set from config
