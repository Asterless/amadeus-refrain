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

QQ chat bot using NoneBot2 + Anthropic Claude API. NapCat handles QQ protocol over WebSocket; NoneBot2 dispatches events to plugins.

```
QQ ←→ NapCat (WS) ←→ NoneBot2 (bot.py)
                        ├── plugins/chat (@bot, priority=10)
                        │     → IdentityManager.resolve()
                        │     → scheduler.interrupt() (cancel pending proactive)
                        │     → LLMClient.chat()
                        │          ├── Anthropic SSE stream
                        │          └── Tool loop (max 5 rounds)
                        │     → scheduler.release()
                        │
                        └── group_listener (priority=1, non-blocking)
                              → GroupTimeline.add()
                              → GroupChatScheduler.notify()
                                   ├── debounce (N sec quiet) → LLM chat
                                   └── batch (M msgs full)    → LLM chat
                                        └── pass_turn tool → skip or reply
```

- Config: `BotConfig` (Pydantic) loaded via `config_loader.py` — TOML < env vars < CLI args
- Ruff: configured in `pyproject.toml`, RUF001/RUF002/RUF003 ignored for Chinese full-width chars
- Docker: **always `docker compose restart napcat`**, never `down` + `up` (device fingerprint → anti-fraud)

Details: [docs/architecture.md](docs/architecture.md) — design decisions, config, tools, access control
Operations: [docs/operations.md](docs/operations.md) — Docker/NapCat, building & updating

## Workflow

All tests must pass (`uv run pytest`) before committing code. Same for lint and type checks.

## Language

User-facing strings and identity configs are in Chinese. Everything else (code, comments, docstrings, log messages) in English.
