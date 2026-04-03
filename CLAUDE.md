# CLAUDE.md

## Commands

```bash
uv sync                        # Install dependencies
uv run ruff check src/         # Lint (add --fix for auto-fix)
uv run pytest                  # Run all tests
uv run pytest tests/test_identity.py::test_name -v  # Single test
uv run pyright                 # Type check
```

| Task | Command |
|------|---------|
| Run locally | `docker compose up napcat -d && uv run python bot.py` |
| Run all in Docker | `docker compose up -d` |
| Restart bot (soul/config) | `docker compose restart bot` |
| Rebuild bot (code/deps) | `docker compose up bot -d --build` |
| Usage TUI | `uv run python -m src.llm.usage_cli tui day\|week\|month [date]` |
| Build & deploy | `./scripts/deploy.sh` |

## Architecture

QQ chat bot: NoneBot2 + Anthropic Claude API. NapCat handles QQ protocol over WebSocket.

```
QQ ←→ NapCat (WS) ←→ NoneBot2 (bot.py)
                        ├── private_chat (DM, priority=10)
                        │     → LLMClient.chat() → Anthropic SSE stream
                        │       └── Tool loop (max 5 rounds), pass_turn to skip
                        └── group_listener (priority=1, non-blocking)
                              → GroupTimeline → GroupChatScheduler
                                ├── @bot → fire immediately
                                ├── debounce (N sec quiet) → LLM chat
                                └── batch (M msgs full) → LLM chat
```

Key design choices:
- **Raw Anthropic API** via aiohttp SSE, no SDK — tool calls touch `_call_api` in `src/llm/client.py`
- **Prompt caching** — system blocks use `cache_control: ephemeral`; second-to-last history msg cached
- **Context compaction** — front half of history compressed via LLM when exceeding `max_context_tokens × compact_ratio`
- **Soul directory** — `soul/identity.md` (persona), `soul/instruction.md` (behavioral directives)
- **Memory** — short-term: in-memory deque per session; long-term: `.qmd` files in `storage/memories/`
- **Config**: `BotConfig` (Pydantic) via `config_loader.py` — TOML < env vars < CLI args
- **Ruff**: `pyproject.toml`, RUF001/RUF002/RUF003 ignored (Chinese full-width chars)
- **Docker**: **always `docker compose restart napcat`**, never `down`+`up` (device fingerprint → anti-fraud)

Deep dives → [docs/architecture.md](docs/architecture.md) | Docker/ops → [docs/operations.md](docs/operations.md)

## Workflow

All tests must pass (`uv run pytest`) before committing. Same for lint and type checks.
Fix any errors discovered during testing, even if they were pre-existing and not introduced by your changes.

## Release

发布新版本时，Docker 镜像版本必须和 git tag 对齐。无正式 tag 时使用 `vYYYYMMDD-<short hash>` 格式（如 `v20260404-cd328d2`）。

## Language

Chinese: user-facing strings, identity configs. English: code, comments, docstrings, logs, commits.

## Maintaining this file

Keep CLAUDE.md as a **self-contained index**: high-frequency knowledge inline, low-frequency details in `docs/`.

- **Inline if**: every task needs it (architecture overview, key patterns, commands, rules)
- **Link if**: only relevant when working on that subsystem (scheduler tuning, Docker build stages, config fields)
- **Add commands** to the table or code block, not as new sections
- **No duplication** with `docs/` — if detail is already linked, don't repeat it here
