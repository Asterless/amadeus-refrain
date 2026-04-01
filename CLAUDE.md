# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run bot locally (needs NapCat running)
docker compose up napcat -d
uv run python bot.py

# Run everything in Docker
docker compose up -d

# Lint
uv run ruff check src/

# Lint with auto-fix
uv run ruff check --fix src/

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_identity.py

# Run a specific test
uv run pytest tests/test_identity.py::test_function_name -v

# Type check
uv run pyright
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
- **Prompt Caching strategy** — System blocks (identity + tool guide, user memory) use `cache_control: ephemeral`. Group chat context goes into messages (not system) because it changes every turn. The second-to-last history message gets `cache_control` to maximize cache hits on conversation prefix.
- **Tool framework** — Tools extend `src/tools/base.py:Tool` ABC (name, description, parameters as JSON Schema, async execute). Registered in `ToolRegistry`, converted to Anthropic format via OpenAI-style intermediate. `ToolContext` carries the Bot instance and event metadata.
- **Identity system** — Parsed from `identities.md` (Markdown with `## id` sections + `- key: value` metadata). Resolution order: manual override > keyword/group match (by priority) > default.
- **Memory layers** — Short-term: in-memory deque per session. Long-term: `.qmd` files per user in `data/memories/` with profile + events sections. Group context: in-memory deque of recent messages per group.
- **Session IDs** — `group_{group_id}` for group chats, `private_{user_id}` for DMs. Used as keys for short-term memory and identity overrides.
- **History bootstrap** — On bot connect, `history_loader.py` pulls recent messages from NapCat HTTP API for all groups, populating both short-term memory and group context.

### Config

All config flows through `src/config.py:BotConfig` (Pydantic model), loaded from `.env` via NoneBot's `get_plugin_config`. NoneBot itself is configured in `pyproject.toml` under `[tool.nonebot]`.

### Ruff

Configured in `pyproject.toml`. RUF001/RUF002/RUF003 are ignored to allow Chinese full-width characters throughout the codebase.

### Docker / NapCat 运维

- NapCat 持久化两个目录：`./napcat/config` (配置) 和 `./napcat/data` (QQ 会话/设备指纹)
- 设备指纹存储在 `napcat/data/nt_qq/global/nt_data/mmkv/`，登录 token 在 `napcat/data/nt_qq/global/nt_data/Login/`
- **重启用 `docker compose restart napcat`**，不要 `down` + `up`（避免设备指纹变化触发风控）
- 掉线通常是腾讯风控，不是持久化问题。风控后 token 服务端失效，需重新登录
- NapCat 使用 NTQQ 协议，支持手机 QQ 同时在线（多设备共存）
- Bot QQ 号：10000（Amadeus），主群：100001

## Language

This is a Chinese-language project. Code comments, docstrings, user-facing strings, and identity configurations are in Chinese.
