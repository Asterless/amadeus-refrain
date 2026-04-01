# CLAUDE.md

## Commands

```bash
uv sync                        # Install dependencies
uv run ruff check src/         # Lint (add --fix for auto-fix)
uv run pytest                  # Run all tests
uv run pytest tests/test_identity.py::test_name -v  # Single test
uv run pyright                 # Type check

# Run bot locally (needs NapCat running)
docker compose up napcat -d && uv run python bot.py

# Run everything in Docker
docker compose up -d
```

## Architecture

QQ chat bot using NoneBot2 framework with Anthropic Claude API. NapCat handles the QQ protocol over WebSocket; NoneBot2 receives events and dispatches to plugins.

### Request Flow

```
QQ ←→ NapCat (WS) ←→ NoneBot2 (bot.py)
                        ├── plugins/chat  →  IdentityManager.resolve()
                        │                 →  PromptBuilder.build_blocks()
                        │                 →  LLMClient.chat()
                        │                      ├── Anthropic SSE stream
                        │                      └── Tool loop (max 5 rounds)
                        └── group_listener (priority=1, non-blocking)
                             → GroupContext.add()
```

### Key Design Decisions

- **Raw Anthropic API via aiohttp SSE** — no SDK. `src/llm/client.py` manually parses SSE `data:` lines to extract text deltas and tool_use blocks. Adding tool calls means touching the `_call_api` function.
- **Prompt Caching strategy** — System blocks (identity + instruction, user memory) use `cache_control: ephemeral`. Group chat context goes into messages (not system) because it changes every turn. The second-to-last history message gets `cache_control` to maximize cache hits on conversation prefix.
- **Tool framework** — Tools extend `src/tools/base.py:Tool` ABC (name, description, parameters as JSON Schema, async execute). Registered in `ToolRegistry`, converted to Anthropic format via OpenAI-style intermediate. `ToolContext` carries the Bot instance and event metadata.
- **Soul directory** — `soul/` holds personality & instruction configs. `identities.md` defines personas (Markdown with `## id` sections + `- key: value` metadata); `instruction.md` holds behavioral directives injected into the system prompt. Resolution order: manual override > keyword/group match (by priority) > default.
- **Memory layers** — Short-term: in-memory deque per session. Long-term: `.qmd` files per user in `storage/memories/` with profile + events sections. Group context: in-memory deque of recent messages per group.
- **Session IDs** — `group_{group_id}` for group chats, `private_{user_id}` for DMs. Used as keys for short-term memory and identity overrides.
- **History bootstrap** — On bot connect, `history_loader.py` pulls recent messages from NapCat HTTP API for all groups, populating both short-term memory and group context.

### Config

All config flows through `src/config.py:BotConfig` (Pydantic model), loaded from `.env` via NoneBot's `get_plugin_config`. NoneBot itself is configured in `pyproject.toml` under `[tool.nonebot]`.

### Ruff

Configured in `pyproject.toml`. RUF001/RUF002/RUF003 are ignored to allow Chinese full-width characters throughout the codebase.

### Docker / NapCat Operations

- NapCat persists two directories: `./napcat/config` (config) and `./napcat/data` (QQ sessions/device fingerprint)
- Device fingerprint at `napcat/data/nt_qq/global/nt_data/mmkv/`, login tokens at `napcat/data/nt_qq/global/nt_data/Login/`
- **Always use `docker compose restart napcat`** — never `down` + `up` (device fingerprint changes trigger Tencent anti-fraud)
- Disconnections are usually Tencent anti-fraud, not persistence issues. Tokens are server-side invalidated; re-login required.
- NapCat uses NTQQ protocol, supports concurrent mobile QQ sessions (multi-device)
- Bot QQ ID: 10000 (Amadeus), main group: 100001

### Building & Updating

`soul/` is volume-mounted (`./soul:/app/soul:ro`). Changes to soul files take effect with a restart:

```bash
docker compose restart bot           # Soul/config changes only
docker compose up bot -d --build     # Code/dependency/Dockerfile changes
```

**Note**: `docker compose restart` does not rebuild images.

## Language

This is a Chinese-language project. Code comments, docstrings, user-facing strings, and identity configurations are in Chinese.
