"""Bot 配置：嵌套 Pydantic 模型，支持 TOML / 环境变量 / CLI 覆盖。"""

from typing import Self

from pydantic import BaseModel, model_validator


class ContextConfig(BaseModel):
    """上下文窗口配置。"""

    max_context_tokens: int = 1_000_000


class UsageConfig(BaseModel):
    """LLM usage tracking configuration."""

    enabled: bool = True
    slow_threshold_s: float = 60.0


class LLMConfig(BaseModel):
    """LLM 接入配置。"""

    base_url: str = "http://127.0.0.1:34567/v1"
    api_key: str = "sk-placeholder"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    context: ContextConfig = ContextConfig()
    usage: UsageConfig = UsageConfig()


class LogConfig(BaseModel):
    """日志配置。"""

    dir: str = "storage/logs"


class SoulConfig(BaseModel):
    """人设与指令配置目录。"""

    dir: str = "soul"


class GroupConfig(BaseModel):
    """群聊上下文配置。"""

    max_timeline_messages: int = 200
    history_load_count: int = 30
    allowed_groups: list[int] = []
    debounce_seconds: float = 5.0
    batch_size: int = 10


class NapcatConfig(BaseModel):
    """NapCat HTTP API 配置。"""

    api_url: str = "http://localhost:29300"


class MemoConfig(BaseModel):
    """备忘录系统配置。"""

    dir: str = "storage/memories"
    user_max_chars: int = 300
    group_max_chars: int = 500
    index_max_lines: int = 200
    history_enabled: bool = True


class CompactConfig(BaseModel):
    """上下文压缩配置。"""

    micro_ratio: float = 0.6
    full_ratio: float = 0.8
    max_failures: int = 3
    cache_hit_warn: float = 90.0

    @model_validator(mode="after")
    def _check_ratios(self) -> Self:
        if self.micro_ratio >= self.full_ratio:
            raise ValueError("micro_ratio must be less than full_ratio")
        return self


class DreamConfig(BaseModel):
    """Dream 整理配置。"""

    enabled: bool = False
    interval_hours: int = 24
    min_compacts: int = 5
    max_rounds: int = 15


class VisionConfig(BaseModel):
    """多模态视觉配置。"""

    enabled: bool = True
    max_images_per_message: int = 5
    max_images_per_request: int = 15
    max_dimension: int = 768
    cache_dir: str = "storage/image_cache"
    cache_max_age_hours: int = 24


class BotConfig(BaseModel):
    """全局 Bot 配置。"""

    llm: LLMConfig = LLMConfig()
    log: LogConfig = LogConfig()
    memo: MemoConfig = MemoConfig()
    compact: CompactConfig = CompactConfig()
    dream: DreamConfig = DreamConfig()
    soul: SoulConfig = SoulConfig()
    group: GroupConfig = GroupConfig()
    napcat: NapcatConfig = NapcatConfig()
    vision: VisionConfig = VisionConfig()
    admins: dict[str, str] = {}
    allowed_private_users: list[int] = []
