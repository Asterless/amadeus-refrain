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
    assert cfg.llm.model == "claude-sonnet-4-6"
    assert cfg.llm.max_tokens == 1024
    assert cfg.llm.context.max_context_tokens == 1_000_000
    assert cfg.compact.micro_ratio == 0.6
    assert cfg.compact.full_ratio == 0.8
    assert cfg.log.dir == "storage/logs"
    assert cfg.soul.dir == "soul"
    assert cfg.group.max_timeline_messages == 200
    assert cfg.group.history_load_count == 30
    assert cfg.group.debounce_seconds == 5.0
    assert cfg.group.batch_size == 10
    assert cfg.napcat.api_url == "http://localhost:29300"
    assert cfg.admins == {}


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
[admins]
"123456" = "管理员A"
"789012" = "管理员B"

[llm]
base_url = "http://custom-llm:8080/v1"
api_key = "sk-test-key"
model = "claude-opus-4"
max_tokens = 2048

[llm.context]
max_context_tokens = 100_000

[compact]
micro_ratio = 0.5
full_ratio = 0.7

[soul]
dir = "custom_soul"

[group]
max_timeline_messages = 100
history_load_count = 50
debounce_seconds = 3.0
batch_size = 5

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
    assert cfg.compact.micro_ratio == 0.5
    assert cfg.compact.full_ratio == 0.7
    assert cfg.soul.dir == "custom_soul"
    assert cfg.group.max_timeline_messages == 100
    assert cfg.group.history_load_count == 50
    assert cfg.group.debounce_seconds == 3.0
    assert cfg.group.batch_size == 5
    assert cfg.napcat.api_url == "http://napcat:29300"
    assert cfg.admins == {"123456": "管理员A", "789012": "管理员B"}


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


def test_usage_config_defaults():
    from src.config import UsageConfig
    cfg = BotConfig()
    assert cfg.llm.usage.enabled is True
    assert cfg.llm.usage.slow_threshold_s == 60.0


def test_memo_config_defaults() -> None:
    from src.config import CompactConfig, DreamConfig, MemoConfig
    m = MemoConfig()
    assert m.dir == "storage/memories"
    assert m.user_max_chars == 300
    assert m.group_max_chars == 500
    assert m.index_max_lines == 200
    assert m.history_enabled is True

    c = CompactConfig()
    assert c.micro_ratio == 0.6
    assert c.full_ratio == 0.8
    assert c.max_failures == 3

    d = DreamConfig()
    assert d.enabled is False
    assert d.interval_hours == 24
    assert d.min_compacts == 5
    assert d.max_rounds == 15


