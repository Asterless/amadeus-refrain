"""Bot 配置：嵌套 Pydantic 模型，支持 TOML / 环境变量 / CLI 覆盖。"""

from pydantic import BaseModel


class CacheConfig(BaseModel):
    """提示缓存预热配置。"""

    warm_enabled: bool = True
    warm_interval_messages: int = 10
    warm_ttl_seconds: int = 300


class ContextConfig(BaseModel):
    """上下文窗口与压缩配置。"""

    max_context_tokens: int = 1_000_000
    compact_ratio: float = 0.7


class LLMConfig(BaseModel):
    """LLM 接入配置。"""

    base_url: str = "http://127.0.0.1:34567/v1"
    api_key: str = "sk-placeholder"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    context: ContextConfig = ContextConfig()
    cache: CacheConfig = CacheConfig()


class LogConfig(BaseModel):
    """日志配置。"""

    dir: str = "storage/logs"


class MemoryConfig(BaseModel):
    """长期记忆存储配置。"""

    dir: str = "storage/memories"


class SoulConfig(BaseModel):
    """人设与指令配置目录。"""

    dir: str = "soul"


class GroupConfig(BaseModel):
    """群聊上下文配置。"""

    max_timeline_messages: int = 200
    history_load_count: int = 30
    allowed_groups: list[int] = []  # 群聊白名单，空 = 不限制


class NapcatConfig(BaseModel):
    """NapCat HTTP API 配置。"""

    api_url: str = "http://localhost:29300"


class BotConfig(BaseModel):
    """全局 Bot 配置。"""

    llm: LLMConfig = LLMConfig()
    log: LogConfig = LogConfig()
    memory: MemoryConfig = MemoryConfig()
    soul: SoulConfig = SoulConfig()
    group: GroupConfig = GroupConfig()
    napcat: NapcatConfig = NapcatConfig()
    superusers: set[str] = set()
    allowed_private_users: list[int] = []  # 私聊白名单，空 = 不限制
