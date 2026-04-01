"""测试 config_loader：TOML 加载、环境变量覆盖、CLI 覆盖。"""

from pathlib import Path

import pytest

from src.config import BotConfig
from src.config_loader import load_config

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------


def test_load_defaults_without_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """没有 TOML 文件时应返回全默认值。"""
    # 确保 BOT_CONFIG_PATH 未设置，且默认 config.toml 不存在
    monkeypatch.delenv("BOT_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    cfg = load_config(config_path=None)

    assert isinstance(cfg, BotConfig)
    assert cfg.llm.base_url == "http://127.0.0.1:34567/v1"
    assert cfg.llm.api_key == "sk-placeholder"
    assert cfg.llm.model == "claude-sonnet-4-20250514"
    assert cfg.llm.max_tokens == 1024
    assert cfg.llm.context.max_context_tokens == 200_000
    assert cfg.llm.context.compact_ratio == 0.7
    assert cfg.llm.cache.warm_enabled is True
    assert cfg.llm.cache.warm_interval_messages == 10
    assert cfg.llm.cache.warm_ttl_seconds == 300
    assert cfg.log.dir == "storage/logs"
    assert cfg.memory.dir == "storage/memories"
    assert cfg.soul.dir == "soul"
    assert cfg.group.max_timeline_messages == 200
    assert cfg.group.history_load_count == 30
    assert cfg.napcat.api_url == "http://localhost:29300"
    assert cfg.superusers == set()


def test_load_from_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TOML 文件中的值应覆盖默认值。"""
    monkeypatch.delenv("BOT_CONFIG_PATH", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("NAPCAT_API_URL", raising=False)

    toml_file = tmp_path / "config.toml"
    _write_toml(
        toml_file,
        """
superusers = ["123456", "789012"]

[llm]
base_url = "http://custom-llm:8080/v1"
api_key = "sk-test-key"
model = "claude-opus-4"
max_tokens = 2048

[llm.context]
max_context_tokens = 100_000
compact_ratio = 0.5

[llm.cache]
warm_enabled = false
warm_interval_messages = 5
warm_ttl_seconds = 600

[memory]
dir = "custom/memories"

[soul]
dir = "custom_soul"

[group]
max_timeline_messages = 100
history_load_count = 50

[napcat]
api_url = "http://napcat:29300"
""",
    )

    cfg = load_config(config_path=str(toml_file))

    assert cfg.llm.base_url == "http://custom-llm:8080/v1"
    assert cfg.llm.api_key == "sk-test-key"
    assert cfg.llm.model == "claude-opus-4"
    assert cfg.llm.max_tokens == 2048
    assert cfg.llm.context.max_context_tokens == 100_000
    assert cfg.llm.context.compact_ratio == 0.5
    assert cfg.llm.cache.warm_enabled is False
    assert cfg.llm.cache.warm_interval_messages == 5
    assert cfg.llm.cache.warm_ttl_seconds == 600
    assert cfg.memory.dir == "custom/memories"
    assert cfg.soul.dir == "custom_soul"
    assert cfg.group.max_timeline_messages == 100
    assert cfg.group.history_load_count == 50
    assert cfg.napcat.api_url == "http://napcat:29300"
    assert cfg.superusers == {"123456", "789012"}


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """环境变量应覆盖 TOML 文件中的值。"""
    toml_file = tmp_path / "config.toml"
    _write_toml(
        toml_file,
        """
[llm]
base_url = "http://from-toml/v1"
api_key = "sk-from-toml"
model = "model-from-toml"

[napcat]
api_url = "http://napcat-from-toml:29300"
""",
    )

    monkeypatch.setenv("LLM_BASE_URL", "http://from-env/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-from-env")
    monkeypatch.setenv("LLM_MODEL", "model-from-env")
    monkeypatch.setenv("NAPCAT_API_URL", "http://napcat-from-env:29300")

    cfg = load_config(config_path=str(toml_file))

    assert cfg.llm.base_url == "http://from-env/v1"
    assert cfg.llm.api_key == "sk-from-env"
    assert cfg.llm.model == "model-from-env"
    assert cfg.napcat.api_url == "http://napcat-from-env:29300"


def test_cli_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 覆盖应优先于环境变量。"""
    monkeypatch.setenv("LLM_BASE_URL", "http://from-env/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-from-env")
    monkeypatch.setenv("LLM_MODEL", "model-from-env")

    cfg = load_config(
        config_path=None,
        cli_overrides={
            "llm_base_url": "http://from-cli/v1",
            "llm_api_key": "sk-from-cli",
            "llm_model": "model-from-cli",
        },
    )

    assert cfg.llm.base_url == "http://from-cli/v1"
    assert cfg.llm.api_key == "sk-from-cli"
    assert cfg.llm.model == "model-from-cli"


def test_bot_config_path_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BOT_CONFIG_PATH 环境变量应指定配置文件路径。"""
    toml_file = tmp_path / "my_config.toml"
    _write_toml(
        toml_file,
        """
[llm]
api_key = "sk-from-bot-config-path"
""",
    )

    monkeypatch.setenv("BOT_CONFIG_PATH", str(toml_file))
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    cfg = load_config(config_path=None)

    assert cfg.llm.api_key == "sk-from-bot-config-path"


def test_default_config_toml_auto_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """当前目录下的 config.toml 应自动加载。"""
    monkeypatch.delenv("BOT_CONFIG_PATH", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    toml_file = tmp_path / "config.toml"
    _write_toml(
        toml_file,
        """
[llm]
model = "auto-detected-model"
""",
    )

    cfg = load_config(config_path=None)

    assert cfg.llm.model == "auto-detected-model"


def test_proactive_config_defaults() -> None:
    from src.config import BotConfig
    config = BotConfig()
    assert config.proactive.model == "claude-haiku-4-5-20251001"
    assert config.proactive.timeout == 3.0
    assert config.proactive.context_lines == 20
    assert config.proactive.cooldown == 60


def test_proactive_config_from_toml(tmp_path: Path) -> None:
    from src.config_loader import load_config
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        '[proactive]\nmodel = "custom-model"\ntimeout = 5.0\ncooldown = 30\n'
    )
    config = load_config(config_path=str(toml_file))
    assert config.proactive.model == "custom-model"
    assert config.proactive.timeout == 5.0
    assert config.proactive.cooldown == 30
