# Docker & Operations

## Docker / NapCat

- NapCat persists two directories: `./napcat/config` (config) and `./napcat/data` (QQ sessions/device fingerprint)
- Device fingerprint at `napcat/data/nt_qq/global/nt_data/mmkv/`, login tokens at `napcat/data/nt_qq/global/nt_data/Login/`
- **Always use `docker compose restart napcat`** — never `down` + `up` (device fingerprint changes trigger Tencent anti-fraud)
- Disconnections are usually Tencent anti-fraud, not persistence issues. Tokens are server-side invalidated; re-login required
- NapCat uses NTQQ protocol, supports concurrent mobile QQ sessions (multi-device)
- Bot QQ ID: 10000 (Amadeus), main group: 100001

## Building & Updating

`soul/` and `.env` are volume-mounted. Changes to soul files take effect with a restart:

```bash
docker compose restart bot           # Soul/config/.env changes only
docker compose up bot -d --build     # Code/dependency/Dockerfile changes
```

**Note**: `docker compose restart` does not rebuild images.

The bot image uses a two-stage Docker build. `GIT_COMMIT` build arg is baked in and logged at startup for version identification.
