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


class ResolvedGroupConfig(BaseModel):
    """resolve() 返回的扁平群配置，所有字段已合并。"""

    blocked_users: set[int] = set()
    at_only: bool = False
    debounce_seconds: float = 5.0
    batch_size: int = 10
    history_load_count: int = 30


class GroupOverride(BaseModel):
    """单个群的覆盖配置，None 表示使用全局值。"""

    blocked_users: list[int] = []
    at_only: bool | None = None
    debounce_seconds: float | None = None
    batch_size: int | None = None
    history_load_count: int | None = None


class GroupConfig(BaseModel):
    """群聊上下文配置。"""

    history_load_count: int = 30
    allowed_groups: list[int] = []
    debounce_seconds: float = 5.0
    batch_size: int = 10
    at_only: bool = False
    blocked_users: list[int] = []
    overrides: dict[int, GroupOverride] = {}

    def resolve(self, group_id: int) -> ResolvedGroupConfig:
        """合并全局默认值与单群覆盖，返回最终生效配置。"""
        base_blocked = set(self.blocked_users)
        override = self.overrides.get(group_id)
        if override is None:
            return ResolvedGroupConfig(
                blocked_users=base_blocked,
                at_only=self.at_only,
                debounce_seconds=self.debounce_seconds,
                batch_size=self.batch_size,
                history_load_count=self.history_load_count,
            )
        o = override
        return ResolvedGroupConfig(
            blocked_users=base_blocked | set(o.blocked_users),
            at_only=o.at_only if o.at_only is not None else self.at_only,
            debounce_seconds=o.debounce_seconds if o.debounce_seconds is not None else self.debounce_seconds,
            batch_size=o.batch_size if o.batch_size is not None else self.batch_size,
            history_load_count=o.history_load_count if o.history_load_count is not None else self.history_load_count,
        )


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

    ratio: float = 0.7
    compress_ratio: float = 0.5
    max_failures: int = 3
    cache_hit_warn: float = 90.0
    cache_alert_window_m: float = 30.0
    cache_alert_cooldown_m: float = 10.0

    @model_validator(mode="after")
    def _check_ratios(self) -> Self:
        if not (0.0 < self.ratio < 1.0):
            raise ValueError("ratio must be between 0 and 1")
        if not (0.0 < self.compress_ratio < 1.0):
            raise ValueError("compress_ratio must be between 0 and 1")
        return self


class DreamConfig(BaseModel):
    """Dream 整理配置。"""

    enabled: bool = False
    interval_hours: int = 24
    max_rounds: int = 15


class VisionConfig(BaseModel):
    """多模态视觉配置。"""

    enabled: bool = True
    max_images_per_message: int = 5
    max_dimension: int = 768
    cache_dir: str = "storage/image_cache"
    cache_max_age_hours: int = 24


class StickerConfig(BaseModel):
    """表情包系统配置。"""

    enabled: bool = True
    storage_dir: str = "storage/stickers"
    max_count: int = 200


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
    sticker: StickerConfig = StickerConfig()
    admins: dict[str, str] = {}
    allowed_private_users: list[int] = []
