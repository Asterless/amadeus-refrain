import argparse
import os
from pathlib import Path

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

from src.config_loader import load_config as _load_config

_bot_config = _load_config(config_path=args.config)
log_dir = Path(_bot_config.log.dir)
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    log_dir / "bot_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="30 days",
    encoding="utf-8",
    level="DEBUG",
)

logger.info("========== Bot 启动配置 ==========")
logger.info(f"LLM model:   {_bot_config.llm.model}")
logger.info(f"LLM base_url: {_bot_config.llm.base_url}")
logger.info(f"LLM max_tokens: {_bot_config.llm.max_tokens}")
logger.info(f"Context max_tokens: {_bot_config.llm.context.max_context_tokens}")
logger.info(f"Memory dir:  {_bot_config.memory.dir}")
logger.info(f"Soul dir:    {_bot_config.soul.dir}")
logger.info(f"Log dir:     {_bot_config.log.dir}")
logger.info(f"NapCat API:  {_bot_config.napcat.api_url}")
logger.info(f"群聊白名单:  {_bot_config.group.allowed_groups or '无限制'}")
logger.info(f"私聊白名单:  {_bot_config.allowed_private_users or '无限制'}")
logger.info("==================================")

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)

nonebot.load_from_toml("pyproject.toml")

if __name__ == "__main__":
    nonebot.run()
