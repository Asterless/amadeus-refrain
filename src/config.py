"""Bot 配置：嵌套 Pydantic 模型，支持 TOML / 环境变量 / CLI 覆盖。"""

from pydantic import BaseModel


class CacheConfig(BaseModel):
    """提示缓存预热配置。"""

    warm_enabled: bool = True
    warm_interval_messages: int = 10
    warm_ttl_seconds: int = 300


class ContextConfig(BaseModel):
    """上下文窗口与压缩配置。"""

    max_context_tokens: int = 200_000
    compact_ratio: float = 0.7


class LLMConfig(BaseModel):
    """LLM 接入配置。"""

    base_url: str = "http://127.0.0.1:34567/v1"
    api_key: str = "sk-placeholder"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    context: ContextConfig = ContextConfig()
    cache: CacheConfig = CacheConfig()


class MemoryConfig(BaseModel):
    """长期记忆存储配置。"""

    dir: str = "data/memories"


class IdentityConfig(BaseModel):
    """身份配置文件路径。"""

    file: str = "identities.md"


class GroupConfig(BaseModel):
    """群聊上下文配置。"""

    max_timeline_messages: int = 200
    history_load_count: int = 30


class NapcatConfig(BaseModel):
    """NapCat HTTP API 配置。"""

    api_url: str = "http://localhost:29300"


class BotConfig(BaseModel):
    """全局 Bot 配置。"""

    llm: LLMConfig = LLMConfig()
    memory: MemoryConfig = MemoryConfig()
    identity: IdentityConfig = IdentityConfig()
    group: GroupConfig = GroupConfig()
    napcat: NapcatConfig = NapcatConfig()
    superusers: set[str] = set()
