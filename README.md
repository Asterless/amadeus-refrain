<div align="center">

# Amadeus in Shell

**一个有人格、有记忆、会主动插话的 QQ 群聊机器人**

基于 NoneBot2 + Anthropic Claude，通过 NapCat 接入 QQ，支持多模态、Tool Calling、长短期记忆与表情包系统。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![NoneBot2](https://img.shields.io/badge/NoneBot2-2.4+-EA5252.svg)](https://nonebot.dev/)
[![OneBot V11](https://img.shields.io/badge/OneBot-V11-black.svg)](https://onebot.dev/)
[![Powered by Claude](https://img.shields.io/badge/Powered%20by-Claude-D97757.svg)](https://www.anthropic.com/claude)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/package_manager-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-D7FF64.svg)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/type%20checked-pyright-1F75CB.svg)](https://github.com/microsoft/pyright)
[![GitHub stars](https://img.shields.io/github/stars/RoggeOhta/amadeus-in-shell?style=social)](https://github.com/RoggeOhta/amadeus-in-shell/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/RoggeOhta/amadeus-in-shell)](https://github.com/RoggeOhta/amadeus-in-shell/commits/master)

</div>

---

## 目录

- [项目亮点](#项目亮点)
- [功能特性](#功能特性)
- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [人设配置](#人设配置)
- [配置参考](#配置参考)
- [Tool Calling 工具集](#tool-calling-工具集)
- [本地开发](#本地开发)
- [项目结构](#项目结构)
- [运维与监控](#运维与监控)
- [技术栈](#技术栈)
- [文档链接](#文档链接)
- [License](#license)

## 项目亮点

- **像真人一样主动发言** — 不是"指令式"机器人，而是按 debounce / batch 节奏主动参与群聊，可在 `soul/identity.md` 里写死插话规则
- **可塑人格 + 长期记忆** — `soul/identity.md` 用 Markdown 定义单一角色与插话规则，`storage/memories/` 持久化用户画像与关键事件，模型可主动 `recall_memo` / `update_memo`
- **多模态视觉** — 自动下载并缩放图片，base64 直送 Anthropic API，支持每条消息图片数限制
- **原生 Anthropic SSE + Prompt Caching** — 不走 SDK，4 个 cache breakpoint 精细控制，群聊场景能稳定打满 90%+ cache hit
- **上下文自动压缩** — 超出阈值自动 LLM 压缩前半段，压缩同时把观察提取进长期记忆，circuit breaker 防失败循环
- **表情包系统** — SHA256 去重的表情包库，模型可以收藏、复用群里看到的表情包
- **完整用量追踪** — SQLite 记录每次 LLM 调用的 token / cache hit / 延迟，自带 Rich TUI 仪表盘 + HTTP API
- **生产可用** — 每群独立配置、SSRF 防护、管理员鉴权、日志切割、Docker 一键部署

## 功能特性

| 类别 | 说明 |
|------|------|
| **LLM 对话** | @机器人或私聊触发，原生 SSE 流式调用，Tool loop 最多 5 轮，`pass_turn` 工具允许模型主动跳过 |
| **主动插话** | 群消息被动收集 → debounce（N 秒静默）/ batch（M 条满）触发 LLM 决策是否发言 |
| **多模态视觉** | 图片自动下载、pyvips 缩放、磁盘缓存，按消息粒度限制图片数 |
| **表情包** | 持久化表情包库（SHA256 去重 + `index.json`），LLM 可收藏 / 发送 / 管理 |
| **短期记忆** | 每会话保留最近 N 轮对话上下文（in-memory deque） |
| **长期记忆** | 用户画像 + 关键事件，`.md` 文件持久化，pending 区由 compaction 自动追加 |
| **群聊时间线** | append-only turns + pending buffer，SQLite 持久化所有原始消息 |
| **Tool Calling** | 网页抓取、DuckDuckGo 搜索、HTTP API、时间查询、记忆读写、表情包工具、群管理 |
| **群管理** | 禁言、设置头衔、主动发群消息（仅管理员可触发的工具调用） |
| **按群配置** | `group.overrides` 支持每群独立覆盖 `at_only` / `debounce` / `batch_size` / `blocked_users` |
| **用量追踪** | SQLite 记录所有 LLM 调用，cache hit 低 / 调用慢自动告警管理员 |
| **Dream Agent** | 后台定期整理待处理记忆、清理表情包库 |
| **SSRF 防护** | 网页抓取工具拒绝内网 / 本机 / 链路本地地址 |

## 架构概览

```
QQ ←→ NapCat (WS) ←→ NoneBot2 (bot.py)
                        ├── private_chat (DM, priority=10)
                        │     → LLMClient.chat() → Anthropic SSE stream
                        │       └── Tool loop (max 5 rounds), pass_turn to skip
                        ├── group_listener (priority=1, non-blocking)
                        │     → GroupTimeline → GroupChatScheduler
                        │       ├── @bot           → fire immediately
                        │       ├── debounce N sec → LLM chat
                        │       └── batch M msgs   → LLM chat
                        └── DreamAgent (background, periodic)
                              → consolidate memos + sticker cleanup
```

**关键设计取舍：**

- **Raw Anthropic API** via aiohttp SSE，不走 SDK — 让 tool 流式细节可控
- **Prompt Caching** — 4 breakpoints：`tools[-1]`、`system[1]` (人设 + instruction)、`system[2]` (索引 + memo)、`messages[near-end]`
- **Context Compaction** — 输入 token 超过 `max_context_tokens × ratio` 时压缩前半段，压缩失败累计触发 circuit breaker 丢弃最旧消息
- **TOML < env vars < CLI args** — `BotConfig` (Pydantic) 三层覆盖

> 想看完整架构和数据流图，请阅读 [docs/architecture.md](docs/architecture.md)

## 快速开始

### 前置条件

- **Docker** & **Docker Compose**
- 一个能访问的 **Anthropic API** 或兼容接口（推荐 Claude Sonnet 4 / Opus）
- 一个 QQ 小号

### 1. Clone 并配置环境

```bash
git clone https://github.com/RoggeOhta/amadeus-in-shell.git
cd amadeus-in-shell
cp .env.example .env
cp config.example.toml config.toml
```

两个文件的分工:

- **`.env`** — NoneBot 框架层(由 `nonebot.init()` 读取)
- **`config.toml`** — bot 业务层(由 `src/config_loader.py` 读取)

最少需要填的字段:

| 文件 | 字段 | 说明 |
|------|------|------|
| `.env` | `SUPERUSERS` | 管理员 QQ 号 JSON 数组,如 `["10001"]` |
| `.env` | `ONEBOT_WS_URLS` | NapCat WebSocket 地址,默认 `["ws://napcat:3001"]` |
| `config.toml` | `[llm].base_url` | Anthropic 兼容 API 地址 |
| `config.toml` | `[llm].api_key` | API Key |
| `config.toml` | `[llm].model` | 模型名,如 `claude-sonnet-4-20250514` |
| `config.toml` | `[admins]` | `"QQ号" = "昵称"` 映射,授权群管理工具 |

`config.toml` 还可调整上下文压缩、debounce、视觉、记忆等高级参数,详见 [配置参考](#配置参考)。LLM 字段也可用 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 环境变量覆盖,优先级 TOML < env < CLI。

### 2. 编辑人设

```bash
$EDITOR soul/identity.md       # 角色定义 + 插话规则
$EDITOR soul/instruction.md    # 行为指令
```

### 3. 一键启动

```bash
docker compose up -d
```

首次启动后访问 [http://localhost:6099](http://localhost:6099)，在 NapCat WebUI 扫码登录 QQ。

### 4. 测试效果

- **私聊** — 直接发消息
- **群聊** — @机器人 + 消息触发，或让 bot 按 `## 插话方式` 规则自己加入对话

## 人设配置

人设通过 `soul/identity.md` 用纯 Markdown 定义：

```markdown
# 角色名

人设描述正文：性格、背景、说话方式...

## 插话方式

主动插话的判断标准：
- 群里讨论 X 话题时插一句
- 看到 Y 类消息时回应
- 不要在 Z 情况下发言
```

| 字段 | 说明 |
|------|------|
| `# 角色名` | 一级标题 = 人设名称（用于自我介绍） |
| 正文 | 性格 / 背景 / 语言风格 |
| `## 插话方式` | 可选，定义主动插话规则；**没有此节就不会主动发言** |

行为级指令（什么能做、什么不能做、回复格式约束等）写在 `soul/instruction.md`。

## 配置参考

`config.toml` 的关键 section：

| Section | 关键字段 | 用途 |
|---------|----------|------|
| `[admins]` | `"qq" = "昵称"` | 注入管理员到系统提示 + 授权群管理工具 |
| `[llm]` | `base_url` / `api_key` / `model` / `max_tokens` | LLM 接入参数 |
| `[llm.context]` | `max_context_tokens` | 上下文窗口大小，超出触发 compact |
| `[llm.usage]` | `enabled` / `slow_threshold_s` | 用量追踪与慢调用告警 |
| `[compact]` | `ratio` / `compress_ratio` / `cache_hit_warn` | 压缩触发比例 + cache 命中率告警 |
| `[memo]` | `dir` / `user_max_chars` / `index_max_lines` | 长期记忆存储 |
| `[group]` | `debounce_seconds` / `batch_size` / `at_only` / `blocked_users` | 群聊全局调度参数 |
| `[group.overrides.<gid>]` | 同 `[group]` 字段子集 | 单群覆盖（未填字段 fallback 全局） |
| `[vision]` | `enabled` / `max_images_per_message` / `max_dimension` | 多模态图片处理 |
| `[dream]` | `enabled` / `interval_hours` / `max_rounds` | Dream Agent 后台整理 |
| `[napcat]` | `api_url` | NapCat HTTP API 地址（用于主动发消息等） |

完整字段列表见 [`config.example.toml`](config.example.toml)。

## Tool Calling 工具集

模型在每次对话中能调用的工具：

| 工具 | 功能 | 备注 |
|------|------|------|
| `recall_memo` | 按 ID 精确查 / 按 query 模糊搜索用户或群的 memo | — |
| `update_memo` | **完整覆写**某条 memo（异步 fire-and-forget） | — |
| `web_fetch` | 抓取网页正文 | 内置 SSRF 防护，拒绝内网 / 本机 / 链路本地地址 |
| `web_search` | DuckDuckGo 搜索 | 默认 5 条，最多 10 条 |
| `http_api` | 调 NapCat HTTP API（取群信息、用户资料等） | — |
| `get_datetime` | 当前日期时间（Asia/Shanghai） | — |
| `save_sticker` | 把群里看到的图存进表情库 | 需要 `sticker.enabled` |
| `send_sticker` | 从表情库选一张发出去 | 需要 `sticker.enabled` |
| `manage_sticker` | 更新表情包描述 / 删除表情包 | 删除仅管理员；需要 `sticker.enabled` |
| `mute_user` | 禁言群成员（duration=0 解除） | 仅管理员可触发 |
| `set_title` | 设置 / 清除群专属头衔 | 仅管理员可触发 |
| `send_group_msg` | 主动向指定群发消息 | 仅管理员可触发 |
| `pass_turn` | 主动跳过本轮回复（不发任何消息） | 由 client 自动注入，所有对话都能用 |

**仅在内部生命周期使用的工具：**

| 工具 | 触发场景 |
|------|----------|
| `append_memo` | 上下文压缩（compact）时由 LLM 自动调用，把新观察追加到 memo 的 `## 待整理` 区 |
| `list_stickers` / `delete_sticker` | Dream Agent 后台运行时使用，整理表情库 |

## 本地开发

```bash
# 安装 uv（Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync

# 仅用 Docker 起 NapCat
docker compose up napcat -d

# 本地跑 Bot（带热重启可改用 nb run --reload）
uv run python bot.py
```

**质量门禁** — 提交前请全部通过：

```bash
uv run ruff check src/        # Lint（加 --fix 自动修复）
uv run pytest                 # 单元测试
uv run pyright                # 类型检查
```

**单测某文件：**

```bash
uv run pytest tests/test_identity.py::test_name -v
```

## 项目结构

```
├── bot.py                      # 入口
├── soul/                       # 人设与指令配置
│   ├── identity.md             # 人设 + 插话规则
│   └── instruction.md          # 行为指令
├── docker-compose.yml          # NapCat + Bot 编排
├── config.example.toml         # 配置示例（覆盖全部字段）
├── src/
│   ├── config.py               # Pydantic 配置模型
│   ├── config_loader.py        # TOML + env vars + CLI 加载
│   ├── plugins/chat/           # NoneBot 对话插件（消息处理入口）
│   ├── identity/               # 人设模型 & 管理器
│   ├── llm/
│   │   ├── client.py           # Anthropic API 客户端（SSE 流式 + Tool loop）
│   │   ├── prompt.py           # 分层 System Prompt + Prompt Caching
│   │   ├── scheduler.py        # 群聊主动插话调度器（debounce/batch）
│   │   ├── dream.py            # Dream Agent
│   │   ├── usage.py            # 用量追踪（SQLite）
│   │   ├── usage_routes.py     # 用量 HTTP API
│   │   └── usage_cli.py        # 用量 TUI 仪表盘
│   ├── memory/
│   │   ├── short_term.py       # 会话级短期记忆
│   │   ├── memo_store.py       # 长期记忆（.md 持久化）
│   │   ├── group_timeline.py   # 群聊时间线（append-only）
│   │   ├── message_log.py      # 群消息 SQLite 持久化
│   │   ├── history_loader.py   # 启动时拉取 NapCat 历史消息
│   │   └── image_cache.py      # 图片下载 / 缩放 / 缓存
│   ├── sticker/store.py        # 表情包库（SHA256 去重）
│   └── tools/                  # Tool Calling 工具集
├── storage/
│   ├── usage.db                # 用量追踪
│   ├── messages.db             # 群消息持久化
│   ├── logs/                   # 日志
│   ├── memories/               # 长期记忆
│   ├── image_cache/            # 图片缓存
│   └── stickers/               # 表情包库
└── tests/
```

## 运维与监控

| 任务 | 命令 |
|------|------|
| 重启 bot（人设/配置变更） | `docker compose restart bot` |
| 重建 bot（代码/依赖变更） | `docker compose up bot -d --build` |
| **重启 NapCat（必须用 restart）** | `docker compose restart napcat` |
| 查看用量 TUI | `uv run python -m src.llm.usage_cli tui day` |
| 一键部署 | `./scripts/deploy.sh` |

**用量 HTTP API：**

- `GET /usage/summary/today` — 今日总览
- `GET /usage/summary/month` — 月度总览
- `GET /usage/top-users` / `/usage/top-groups`
- `GET /usage/timeseries`

> **重要：** NapCat 容器只能用 `restart`，**不要** `down` + `up`。device fingerprint 会变，触发 QQ 风控。

更多运维细节见 [docs/operations.md](docs/operations.md)。

## 技术栈

- **Python 3.12** + [uv](https://github.com/astral-sh/uv)
- [NoneBot2](https://nonebot.dev/) — 异步机器人框架
- [OneBot V11](https://onebot.dev/) — QQ 协议适配
- [NapCat](https://napneko.github.io/) — QQ 协议实现（Docker）
- **Anthropic Claude API** — 原生 SSE 流式 + Prompt Caching
- [Pydantic](https://docs.pydantic.dev/) — 配置与数据模型
- [aiohttp](https://docs.aiohttp.org/) / [httpx](https://www.python-httpx.org/) — 异步 HTTP
- [aiosqlite](https://github.com/omnilib/aiosqlite) — 异步 SQLite
- [pyvips](https://github.com/libvips/pyvips) — 图片处理
- [Rich](https://github.com/Textualize/rich) — TUI 仪表盘
- [Ruff](https://github.com/astral-sh/ruff) + [Pyright](https://github.com/microsoft/pyright) — 代码质量

## 文档链接

- [架构深挖](docs/architecture.md) — 数据流、调度器、压缩策略细节
- [运维手册](docs/operations.md) — Docker / 升级 / 故障排查
- [CLAUDE.md](CLAUDE.md) — 给 Claude Code 等 AI 协作的项目索引

## License

[MIT](LICENSE) © 2025 RoggeOhta

---

<div align="center">

如果这个项目对你有帮助，欢迎 Star

</div>
