"""Bot 配置：从 .env 加载，通过 NoneBot get_plugin_config 注入。"""

from pydantic import BaseModel


class BotConfig(BaseModel):
    # LLM
    llm_base_url: str = "http://127.0.0.1:34567/v1"
    llm_api_key: str = "sk-placeholder"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_context_tokens: int = 200_000
    compact_ratio: float = 0.7

    # Memory
    memory_dir: str = "data/memories"

    # Identity
    identities_file: str = "identities.md"

    # NapCat HTTP API
    napcat_api_url: str = "http://localhost:29300"

    # Superusers (for admin tool authorization)
    superusers: set[str] = set()
