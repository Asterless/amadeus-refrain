# QQ LLM Bot

基于 NoneBot2 + Anthropic Claude 的 QQ 群聊机器人，通过 NapCat 接入 QQ，支持多人设切换、长短期记忆、Tool Calling 和群管理。

## 功能

- **LLM 对话** — @机器人触发，流式调用 Anthropic API，支持 Prompt Caching
- **多人设系统** — Markdown 配置人设，按关键词 / 群号自动匹配或手动切换
- **短期记忆** — 每个会话保留最近 N 轮对话上下文
- **长期记忆** — 用户画像 + 关键事件，`.qmd` 文件持久化，LLM 主动读写
- **群聊上下文** — 被动收集群内消息，让 Bot 了解聊天语境
- **Tool Calling** — 网页抓取、HTTP API 调用、时间查询、记忆管理
- **群管理工具** — 禁言、设置头衔、主动发群消息（SUPERUSER 鉴权）
- **SSRF 防护** — 网页抓取工具拒绝内网/本机地址

## 架构

```
NapCat (QQ 协议) ←ws→ NoneBot2 (bot.py)
                          ├── plugins/chat    ← 对话入口 (@机器人)
                          ├── identity/       ← 人设管理
                          ├── llm/            ← Anthropic API 客户端 + Prompt 构建
                          ├── memory/         ← 短期 / 长期 / 群聊上下文
                          └── tools/          ← Tool Calling 框架
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
- `/identity list` — 查看可用人设
- `/identity <id>` — 切换人设
- `/identity reset` — 恢复自动匹配

## 人设配置

编辑 `identities.md`，使用 Markdown 格式定义人设：

```markdown
## catgirl

- name: 猫娘
- priority: 10
- keywords: 小喵, 猫娘
- groups: 123456

人设描述正文...
```

| 字段 | 说明 |
|------|------|
| `name` | 人设名称 |
| `priority` | 优先级，数字越大越优先 |
| `keywords` | 触发关键词，消息包含则激活 |
| `groups` | 适用群号，逗号分隔 |

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
```

## 项目结构

```
├── bot.py                  # 入口
├── identities.md           # 人设配置
├── docker-compose.yml      # NapCat + Bot 编排
├── src/
│   ├── config.py           # Pydantic 配置
│   ├── plugins/chat/       # NoneBot 对话插件
│   ├── identity/           # 人设模型 & 管理器
│   ├── llm/
│   │   ├── client.py       # Anthropic API 客户端 (SSE 流式)
│   │   └── prompt.py       # 分层 System Prompt 构建
│   ├── memory/
│   │   ├── short_term.py   # 会话级短期记忆
│   │   ├── long_term.py    # 用户画像 + 事件持久化
│   │   └── group_context.py # 群聊消息上下文
│   └── tools/              # Tool Calling 工具集
│       ├── web_fetch.py    # 网页抓取 (带 SSRF 防护)
│       ├── memory_tool.py  # 记忆读写
│       ├── group_admin.py  # 群管理 (禁言/头衔/发消息)
│       ├── datetime_tool.py
│       └── http_api.py     # 通用 HTTP 调用
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

## License

MIT
