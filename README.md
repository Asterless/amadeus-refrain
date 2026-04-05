# QQ LLM Bot

基于 NoneBot2 + Anthropic Claude 的 QQ 群聊机器人，通过 NapCat 接入 QQ，支持主动插话、长短期记忆、多模态图片理解、表情包收藏、Tool Calling 和群管理。

## 功能

- **LLM 对话** — @机器人或私聊触发，流式调用 Anthropic API，支持 Prompt Caching
- **主动插话** — 被动收集群消息，debounce/batch 触发 LLM 决策是否发言
- **多模态视觉** — 图片自动下载、缩放、缓存，以 base64 发送给 LLM
- **表情包系统** — SHA256 去重的持久化表情包库，LLM 可收藏/发送表情包
- **短期记忆** — 每个会话保留最近 N 轮对话上下文
- **长期记忆** — 用户画像 + 关键事件，`.md` 文件持久化，LLM 主动读写
- **群聊上下文** — append-only 时间线 + SQLite 消息持久化，上下文压缩时提取记忆
- **Tool Calling** — 网页抓取、搜索、HTTP API 调用、时间查询、记忆管理
- **群管理工具** — 禁言、设置头衔、主动发群消息（管理员鉴权）
- **按群配置** — 每个群可独立覆盖 debounce、batch_size、屏蔽用户等参数
- **用量追踪** — SQLite 记录所有 LLM 调用，Rich TUI 仪表盘 + HTTP API
- **Dream Agent** — 后台定期整理待处理记忆、清理表情包库
- **SSRF 防护** — 网页抓取工具拒绝内网/本机地址

## 架构

```
NapCat (QQ 协议) ←ws→ NoneBot2 (bot.py)
                        ├── private_chat (DM, priority=10)
                        │     → LLMClient.chat() → Anthropic SSE stream
                        │       └── Tool loop (max 5 rounds), pass_turn to skip
                        ├── group_listener (priority=1, non-blocking)
                        │     → GroupTimeline → GroupChatScheduler
                        │       ├── @bot → fire immediately
                        │       ├── debounce (N sec quiet) → LLM chat
                        │       └── batch (M msgs full) → LLM chat
                        └── DreamAgent (background, periodic)
                              → consolidate memos + sticker cleanup
```

## 快速开始

### 前置条件

- Docker & Docker Compose
- 可用的 Anthropic API（或兼容接口）

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入：

| 变量 | 说明 |
|------|------|
| `SUPERUSERS` | 管理员 QQ 号，JSON 数组 |
| `ONEBOT_WS_URLS` | NapCat WebSocket 地址 |
| `LLM_BASE_URL` | LLM API 地址 |
| `LLM_API_KEY` | API Key |
| `LLM_MODEL` | 模型名称 |

### 2. 启动

```bash
docker compose up -d
```

首次启动后访问 `http://localhost:6099` 在 NapCat WebUI 中扫码登录 QQ。

### 3. 使用

- **@机器人 + 消息** — 触发对话
- 机器人也会根据 `soul/identity.md` 中的 `## 插话方式` 规则主动参与群聊

## 人设配置

编辑 `soul/identity.md`，使用 Markdown 格式定义人设：

```markdown
# 角色名

人设描述正文...

## 插话方式

主动插话的规则和判断标准...
```

- `# 角色名` — 一级标题定义人设名称
- 正文 — 人设性格、背景描述
- `## 插话方式` — 可选，定义主动插话规则（无此节则不会主动发言）

行为指令在 `soul/instruction.md` 中配置。

## 本地开发

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync

# 仅启动 NapCat
docker compose up napcat -d

# 运行 Bot
cp .env.example .env  # 编辑配置
uv run python bot.py

# 代码检查 & 测试
uv run ruff check src/
uv run pytest
uv run pyright
```

## 项目结构

```
├── bot.py                      # 入口
├── soul/                       # 人设与指令配置
│   ├── identity.md             # 人设定义
│   └── instruction.md          # 行为指令
├── docker-compose.yml          # NapCat + Bot 编排
├── src/
│   ├── config.py               # Pydantic 配置模型
│   ├── config_loader.py        # TOML + 环境变量 + CLI 参数加载
│   ├── plugins/chat/           # NoneBot 对话插件（消息处理入口）
│   ├── identity/               # 人设模型 & 管理器
│   ├── llm/
│   │   ├── client.py           # Anthropic API 客户端 (SSE 流式 + Tool loop)
│   │   ├── prompt.py           # 分层 System Prompt 构建 + Prompt Caching
│   │   ├── scheduler.py        # 群聊主动插话调度器 (debounce/batch)
│   │   ├── dream.py            # Dream Agent (后台定期整理记忆)
│   │   ├── usage.py            # 用量追踪 (SQLite)
│   │   ├── usage_routes.py     # 用量 HTTP API
│   │   └── usage_cli.py        # 用量 TUI 仪表盘入口
│   ├── memory/
│   │   ├── short_term.py       # 会话级短期记忆 (deque)
│   │   ├── memo_store.py       # 长期记忆 (.md 文件持久化)
│   │   ├── group_timeline.py   # 群聊时间线 (append-only turns + pending)
│   │   ├── message_log.py      # 群消息 SQLite 持久化
│   │   ├── history_loader.py   # 启动时从 NapCat 加载历史消息
│   │   ├── image_cache.py      # 图片下载、缩放、缓存
│   │   └── types.py            # 消息内容类型定义
│   ├── sticker/
│   │   └── store.py            # 表情包库 (SHA256 去重 + index.json)
│   ├── tools/                  # Tool Calling 工具集
│   │   ├── base.py             # Tool ABC
│   │   ├── registry.py         # 工具注册 & 格式转换
│   │   ├── context.py          # 工具执行上下文
│   │   ├── memo_tools.py       # 记忆读写 (recall_memo, update_memo)
│   │   ├── web_fetch.py        # 网页抓取 (带 SSRF 防护)
│   │   ├── web_search.py       # DuckDuckGo 搜索
│   │   ├── http_api.py         # NapCat HTTP API 调用
│   │   ├── group_admin.py      # 群管理 (禁言/头衔/发消息)
│   │   ├── sticker_tools.py    # 表情包工具 (收藏/发送/管理)
│   │   └── datetime_tool.py    # 时间查询
│   └── constants/
│       └── qq_face.py          # QQ 表情映射
├── storage/
│   ├── usage.db                # 用量追踪数据库
│   ├── messages.db             # 群消息持久化数据库
│   ├── logs/                   # 日志文件
│   ├── memories/               # 长期记忆 (.md)
│   ├── image_cache/            # 图片缓存
│   └── stickers/               # 表情包库
└── tests/
```

## 技术栈

- **Python 3.12** + [uv](https://github.com/astral-sh/uv)
- **NoneBot2** — 异步机器人框架
- **OneBot V11** — QQ 协议适配
- **NapCat** — QQ 协议实现（Docker）
- **Anthropic API** — LLM 对话（原生 SSE 流式 + Prompt Caching）
- **Pydantic** — 配置与数据模型
- **aiohttp / httpx** — 异步 HTTP
- **aiosqlite** — 异步 SQLite（用量追踪 + 消息持久化）
- **pyvips** — 图片缩放处理
- **Rich** — TUI 仪表盘

## License

MIT
