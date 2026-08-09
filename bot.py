from __future__ import annotations

import argparse
import os
from contextlib import suppress
from pathlib import Path

import loguru
import nonebot
from loguru import logger
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

parser = argparse.ArgumentParser(description="QQ Bot")
parser.add_argument("--config", default=None, help="配置文件路径")
parser.add_argument("--llm-base-url", default=None)
parser.add_argument("--llm-api-key", default=None)
parser.add_argument("--llm-model", default=None)
args = parser.parse_args()

if args.config:
    os.environ["BOT_CONFIG_PATH"] = args.config
if args.llm_base_url:
    os.environ["_CLI_LLM_BASE_URL"] = args.llm_base_url
if args.llm_api_key:
    os.environ["_CLI_LLM_API_KEY"] = args.llm_api_key
if args.llm_model:
    os.environ["_CLI_LLM_MODEL"] = args.llm_model

from src.config_loader import load_config as _load_config  # noqa: E402

_bot_config = _load_config(config_path=args.config)
log_dir = Path(_bot_config.log.dir)
log_dir.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(
    log_dir / "bot_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="30 days",
    encoding="utf-8",
    level="DEBUG",
    filter=lambda record: not record["extra"].get("dream", False),
)

_c = _bot_config
logger.info("========== Bot 启动 ({}) ==========", os.environ.get("GIT_COMMIT", "dev"))
logger.info(
    "[LLM] model={} base_url={} max_tokens={}",
    _c.llm.model, _c.llm.base_url, _c.llm.max_tokens,
)
logger.info(
    "[Context] max_tokens={} compact_ratio={} compress_ratio={}",
    _c.llm.context.max_context_tokens, _c.compact.ratio, _c.compact.compress_ratio,
)
logger.info(
    "[Group] debounce={:.1f}s batch_size={}",
    _c.group.debounce_seconds, _c.group.batch_size,
)
logger.info(
    "[Group] history_load={} allowed={}",
    _c.group.history_load_count,
    _c.group.allowed_groups or "无限制",
)
logger.info(
    "[Access] admins={} private_whitelist={}",
    _c.admins or "无", _c.allowed_private_users or "无限制",
)
logger.info(
    "[Dirs] soul={} memo={} log={}",
    _c.soul.dir, _c.memo.dir, _c.log.dir,
)
logger.info("[NapCat] api_url={}", _c.napcat.api_url)
logger.info("==================================")

nonebot.init()

# Suppress NoneBot per-message matcher noise while keeping our own INFO logs.
# Replace NoneBot's stderr handler with one that filters matcher handled/complete.
import nonebot.log as _nlog  # noqa: E402

_MATCHER_NOISE = ("Event will be handled by", "running complete")


def _quiet_filter(record: loguru.Record) -> bool:
    if not _nlog.default_filter(record):  # type: ignore[arg-type]
        return False
    name = record.get("name") or ""
    return not (name.startswith("nonebot") and any(s in record["message"] for s in _MATCHER_NOISE))


if hasattr(_nlog, "logger_id"):
    with suppress(ValueError):
        logger.remove(_nlog.logger_id)
    _nlog.logger_id = logger.add(  # type: ignore[assignment]
        __import__("sys").stderr,
        level=0,
        diagnose=False,
        filter=_quiet_filter,
        format=_nlog.default_format,
    )

driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)

nonebot.load_from_toml("pyproject.toml")

if __name__ == "__main__":
    nonebot.run()
