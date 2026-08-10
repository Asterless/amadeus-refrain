"""Bot 配置：嵌套 Pydantic 模型，支持 TOML / 环境变量 / CLI 覆盖。"""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


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
    proactive_cooldown_seconds: float = 60.0
    proactive_max_replies_per_hour: int = 12


class GroupOverride(BaseModel):
    """单个群的覆盖配置，None 表示使用全局值。"""

    blocked_users: list[int] = []
    at_only: bool | None = None
    debounce_seconds: float | None = None
    batch_size: int | None = None
    history_load_count: int | None = None
    proactive_cooldown_seconds: float | None = None
    proactive_max_replies_per_hour: int | None = None


class GroupConfig(BaseModel):
    """群聊上下文配置。"""

    history_load_count: int = 30
    allowed_groups: list[int] = []
    debounce_seconds: float = 5.0
    batch_size: int = 10
    at_only: bool = False
    startup_catchup: bool = False
    blocked_users: list[int] = []
    proactive_cooldown_seconds: float = 60.0
    proactive_max_replies_per_hour: int = 12
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
                proactive_cooldown_seconds=self.proactive_cooldown_seconds,
                proactive_max_replies_per_hour=self.proactive_max_replies_per_hour,
            )
        o = override
        return ResolvedGroupConfig(
            blocked_users=base_blocked | set(o.blocked_users),
            at_only=o.at_only if o.at_only is not None else self.at_only,
            debounce_seconds=o.debounce_seconds if o.debounce_seconds is not None else self.debounce_seconds,
            batch_size=o.batch_size if o.batch_size is not None else self.batch_size,
            history_load_count=o.history_load_count if o.history_load_count is not None else self.history_load_count,
            proactive_cooldown_seconds=(
                o.proactive_cooldown_seconds
                if o.proactive_cooldown_seconds is not None
                else self.proactive_cooldown_seconds
            ),
            proactive_max_replies_per_hour=(
                o.proactive_max_replies_per_hour
                if o.proactive_max_replies_per_hour is not None
                else self.proactive_max_replies_per_hour
            ),
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


class MemeConfig(BaseModel):
    """Realtime trend discovery and meme verification."""

    enabled: bool = True
    hotboard_url: str = "https://uapis.cn/api/v1/misc/hotboard"
    platforms: list[str] = ["weibo", "bilibili", "douyin", "xiaohongshu", "zhihu", "baidu"]
    refresh_minutes: int = Field(default=15, ge=1)
    per_platform_limit: int = Field(default=30, ge=1, le=50)
    storage_file: str = "storage/memes.json"
    knowledge_file: str = "storage/meme_knowledge.db"
    active_hours: int = Field(default=72, ge=1)
    max_entries: int = Field(default=500, ge=20)
    max_prompt_entries: int = Field(default=12, ge=1, le=30)


class MusicConfig(BaseModel):
    """NetEase Cloud Music API and login session settings."""

    enabled: bool = True
    api_base_url: str = "http://127.0.0.1:3000"
    cookie_file: str = "storage/netease_cookie.json"
    timeout_seconds: float = Field(default=15.0, ge=2, le=60)
    auto_start: bool = False
    service_app: str = ""
    node_executable: str = "node"


class TTSConfig(BaseModel):
    """Text-to-speech provider settings."""

    enabled: bool = True
    provider: Literal["edge", "gpt_sovits"] = "edge"
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "+0%"
    volume: str = "+0%"
    proxy: str = ""
    base_url: str = "http://host.docker.internal:9880"
    ref_audio_path: str = ""
    prompt_text: str = ""
    prompt_lang: str = "zh"
    text_lang: str = "zh"
    text_split_method: str = "cut5"
    media_type: Literal["wav", "ogg", "aac"] = "wav"
    timeout_seconds: float = Field(default=120.0, ge=5, le=600)
    max_chars: int = Field(default=300, ge=1, le=1000)


class SearchConfig(BaseModel):
    """Optional OpenAI native web search augmentation."""

    openai_enabled: bool = False
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"


class VisionConfig(BaseModel):
    """多模态视觉配置。"""

    enabled: bool = True
    max_images_per_message: int = 5
    max_dimension: int = 768
    cache_dir: str = "storage/image_cache"
    cache_max_age_hours: int = 24
    # 免费识图预处理（OpenAI 兼容端点，如智谱 GLM-4.6V-Flash）
    # 聊天模型不支持图片时，先用这个模型把图转成文字描述
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    # 识图策略：only_at = 仅被 @ 触发回复时识图（省免费额度）；always = 所有回复都识图
    describe_mode: Literal["only_at", "always"] = "only_at"


class StickerConfig(BaseModel):
    """表情包系统配置。"""

    enabled: bool = True
    reply_on_receive: bool = False
    send_probability: float = Field(default=0.8, ge=0.0, le=1.0)
    storage_dir: str = "storage/stickers"
    max_count: int = 200
    auto_collect: bool = True
    auto_collect_only_stickers: bool = True
    auto_collect_cooldown_seconds: int = 8


class BotConfig(BaseModel):
    """全局 Bot 配置。"""

    llm: LLMConfig = LLMConfig()
    log: LogConfig = LogConfig()
    memo: MemoConfig = MemoConfig()
    compact: CompactConfig = CompactConfig()
    dream: DreamConfig = DreamConfig()
    meme: MemeConfig = MemeConfig()
    music: MusicConfig = MusicConfig()
    tts: TTSConfig = TTSConfig()
    search: SearchConfig = SearchConfig()
    soul: SoulConfig = SoulConfig()
    group: GroupConfig = GroupConfig()
    napcat: NapcatConfig = NapcatConfig()
    vision: VisionConfig = VisionConfig()
    sticker: StickerConfig = StickerConfig()
    admins: dict[str, str] = {}
    allowed_private_users: list[int] = []
