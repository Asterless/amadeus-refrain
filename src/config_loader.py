"""配置加载器：TOML → 环境变量 → CLI 参数，按优先级依次覆盖。"""

import os
import tomllib
from typing import Any

from src.config import BotConfig

# ---------------------------------------------------------------------------
# 映射表：环境变量名 → dotted key（嵌套路径）
# ---------------------------------------------------------------------------

_ENV_MAP: dict[str, str] = {
    "LLM_BASE_URL": "llm.base_url",
    "LLM_API_KEY": "llm.api_key",
    "LLM_MODEL": "llm.model",
    "NAPCAT_API_URL": "napcat.api_url",
    "VISION_BASE_URL": "vision.base_url",
    "VISION_API_KEY": "vision.api_key",
    "VISION_MODEL": "vision.model",
    "SEARCH_OPENAI_ENABLED": "search.openai_enabled",
    "SEARCH_OPENAI_BASE_URL": "search.openai_base_url",
    "SEARCH_OPENAI_API_KEY": "search.openai_api_key",
    "SEARCH_OPENAI_MODEL": "search.openai_model",
}

# CLI 参数名 → dotted key（由 bot.py argparse 写入环境变量 _CLI_* 或直接传入 cli_overrides）
_CLI_MAP: dict[str, str] = {
    "llm_base_url": "llm.base_url",
    "llm_api_key": "llm.api_key",
    "llm_model": "llm.model",
}


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _deep_set(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """将 dotted_key（如 'llm.base_url'）写入嵌套字典 d。"""
    keys = dotted_key.split(".")
    node = d
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def load_config(
    config_path: str | None = None,
    cli_overrides: dict[str, str] | None = None,
) -> BotConfig:
    """加载并合并配置。

    优先级（低 → 高）：
      1. Pydantic 默认值
      2. TOML 文件
      3. 环境变量（_ENV_MAP）
      4. _CLI_* 环境变量（由 bot.py argparse 设置）
      5. cli_overrides 参数

    Args:
        config_path: TOML 文件路径。为 None 时依次检查 BOT_CONFIG_PATH 环境变量、
                     当前目录的 config.toml。
        cli_overrides: 从命令行参数解析得到的覆盖字典（key 为 _CLI_MAP 中的名称）。
    """
    data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. 确定 TOML 文件路径
    # ------------------------------------------------------------------
    resolved_path: str | None = config_path
    if resolved_path is None:
        resolved_path = os.environ.get("BOT_CONFIG_PATH")
    if resolved_path is None:
        default_toml = "config.toml"
        if os.path.isfile(default_toml):
            resolved_path = default_toml

    # ------------------------------------------------------------------
    # 2. 加载 TOML
    # ------------------------------------------------------------------
    if resolved_path is not None:
        with open(resolved_path, "rb") as fh:
            data = tomllib.load(fh)

    # ------------------------------------------------------------------
    # 3. 环境变量覆盖
    # ------------------------------------------------------------------
    for env_var, dotted_key in _ENV_MAP.items():
        value = os.environ.get(env_var)
        if value is not None:
            _deep_set(data, dotted_key, value)

    # ------------------------------------------------------------------
    # 4. _CLI_* 环境变量覆盖（由 bot.py 写入，供跨进程传递 CLI 参数）
    # ------------------------------------------------------------------
    for cli_key, dotted_key in _CLI_MAP.items():
        env_name = f"_CLI_{cli_key.upper()}"
        value = os.environ.get(env_name)
        if value is not None:
            _deep_set(data, dotted_key, value)

    # ------------------------------------------------------------------
    # 5. cli_overrides 参数覆盖
    # ------------------------------------------------------------------
    if cli_overrides:
        for cli_key, value in cli_overrides.items():
            dotted_key = _CLI_MAP.get(cli_key)
            if dotted_key is None:
                # 也允许直接使用 dotted key
                dotted_key = cli_key
            _deep_set(data, dotted_key, value)

    return BotConfig.model_validate(data)
