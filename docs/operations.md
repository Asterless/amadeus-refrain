# Docker & Operations

## Docker / NapCat

- NapCat persists two directories: `./napcat/config` (config) and `./napcat/data` (QQ sessions/device fingerprint)
- Device fingerprint at `napcat/data/nt_qq/global/nt_data/mmkv/`, login tokens at `napcat/data/nt_qq/global/nt_data/Login/`
- **Always use `docker compose restart napcat`** — never `down` + `up` (device fingerprint changes trigger Tencent anti-fraud)
- Disconnections are usually Tencent anti-fraud, not persistence issues. Tokens are server-side invalidated; re-login required
- NapCat uses NTQQ protocol, supports concurrent mobile QQ sessions (multi-device)

## Building & Updating

`soul/` and `.env` are volume-mounted. Changes to soul files take effect with a restart:

```bash
docker compose restart bot           # Soul/config/.env changes only
docker compose up bot -d --build     # Code/dependency/Dockerfile changes
```

**Note**: `docker compose restart` does not rebuild images.

The bot image uses a two-stage Docker build. `GIT_COMMIT` build arg is baked in and logged at startup for version identification.

## Deploy Script

`scripts/deploy.sh` automates building and deploying:

1. Auto-detects version: uses git tag if present, otherwise `vYYYYMMDD-{short_hash}`
2. Builds bot image with `GIT_COMMIT` build arg
3. Tags image as `qq-bot:{version}` and `qq-bot:latest`
4. Runs `docker compose up bot -d`

This script only targets the bot service. For a full first-time deployment, see [docs/deployment.md](docs/deployment.md). NapCat should already be running or started separately with `docker compose up -d`.

## Storage Layout

```
storage/
├── usage.db                    # SQLite — LLM usage tracking
├── messages.db                 # SQLite — raw group message persistence
├── logs/
│   ├── bot_*.log               # Main bot logs (10MB rotation, 30 days retention)
│   └── dream_*.log             # Dream agent logs (separate sink)
├── memories/
│   ├── users/                  # Per-user .md memo files
│   ├── groups/                 # Per-group .md memo files
│   └── index.md                # Cross-reference index
├── image_cache/
│   └── {2-char-bucket}/        # Cached images (auto-cleanup on startup)
│       └── {file_id}.jpg
└── stickers/
    ├── index.json              # Sticker metadata & usage stats
    └── stk_{hash}.{ext}        # Sticker image files
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `nonebot2[fastapi]` | Bot framework + HTTP server |
| `nonebot-adapter-onebot` | OneBot V11 protocol adapter |
| `aiohttp` | Anthropic API SSE streaming |
| `pydantic` | Config validation |
| `aiosqlite` | Usage tracking SQLite async |
| `rich` | Usage TUI dashboard |
| `pyvips` | Image downscaling (requires libvips system lib) |
| `duckduckgo-search` | Web search tool backend |
| `aiofiles` | Async file I/O |
| `tenacity` | Retry logic |
| `loguru` | Structured logging |
| `httpx` | HTTP client (NapCat API) |

## Usage Monitoring

**TUI**: `uv run python -m src.llm.usage_cli tui day|week|month [date]` — interactive Rich dashboard showing token usage, cache hit rates, call breakdowns.

**API**: when `llm.usage.enabled = true`, FastAPI routes are mounted:

| Endpoint | Description |
|----------|-------------|
| `GET /usage/summary/today` | Today's usage summary |
| `GET /usage/summary/month` | Current month's summary |
| `GET /usage/top-users` | Top users by token usage |
| `GET /usage/top-groups` | Top groups by token usage |
| `GET /usage/timeseries` | Hourly token breakdown |

**Alerts**: when enabled, bot PMs all configured `admins` when:
- Average cache hit rate drops below `compact.cache_hit_warn` % over a rolling window
- A single LLM call exceeds `llm.usage.slow_threshold_s` seconds
