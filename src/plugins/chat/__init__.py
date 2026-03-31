"""对话插件：@机器人 触发，Soul + 记忆 + 工具 + 多人设 + 群聊上下文。"""

from loguru import logger
from nonebot import get_driver, get_plugin_config, on_command, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageEvent
from nonebot.params import CommandArg
from nonebot.rule import to_me

from src.config import BotConfig
from src.identity import IdentityManager
from src.llm.client import LLMClient
from src.llm.prompt import PromptBuilder
from src.memory.group_context import GroupContext
from src.memory.history_loader import load_group_history
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.tools import ToolRegistry
from src.tools.context import ToolContext
from src.tools.datetime_tool import DateTimeTool
from src.tools.group_admin import MuteUserTool, SendGroupMsgTool, SetTitleTool
from src.tools.http_api import HttpApiTool
from src.tools.memory_tool import RecallMemoryTool, SaveMemoryTool
from src.tools.web_fetch import WebFetchTool

driver = get_driver()
bot_config = get_plugin_config(BotConfig)

_llm: LLMClient
_identity_mgr: IdentityManager
_group_ctx: GroupContext
_short_term: ShortTermMemory


@driver.on_startup
async def _init() -> None:
    global _llm, _identity_mgr, _group_ctx, _short_term

    long_term = LongTermMemory(memory_dir=bot_config.memory_dir)
    _short_term = ShortTermMemory()
    _group_ctx = GroupContext()
    prompt_builder = PromptBuilder(long_term=long_term, group_context=_group_ctx)
    short_term = _short_term

    superusers = bot_config.superusers | driver.config.superusers

    tools = ToolRegistry()
    tools.register(SaveMemoryTool(long_term))
    tools.register(RecallMemoryTool(long_term))
    tools.register(DateTimeTool())
    tools.register(WebFetchTool())
    tools.register(HttpApiTool())
    tools.register(MuteUserTool(superusers))
    tools.register(SetTitleTool(superusers))
    tools.register(SendGroupMsgTool(superusers))

    _identity_mgr = IdentityManager()
    await _identity_mgr.load_file(bot_config.identities_file)

    _llm = LLMClient(
        base_url=bot_config.llm_base_url,
        api_key=bot_config.llm_api_key,
        model=bot_config.llm_model,
        prompt_builder=prompt_builder,
        short_term=short_term,
        tools=tools,
        max_context_tokens=bot_config.llm_max_context_tokens,
        compact_ratio=bot_config.compact_ratio,
    )


@driver.on_shutdown
async def _shutdown() -> None:
    await _llm.close()


@driver.on_bot_connect
async def _on_connect(bot: Bot) -> None:
    """Bot 连接后拉取群历史消息，填充群聊上下文。"""
    try:
        group_list: list[dict[str, object]] = await bot.get_group_list()
        group_ids = [str(g["group_id"]) for g in group_list]
        logger.info("loading history | groups={}", len(group_ids))
        await load_group_history(
            napcat_url=bot_config.napcat_api_url,
            group_ids=group_ids,
            group_context=_group_ctx,
        )
    except Exception:
        logger.exception("failed to load group history")


def _session_id(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"group_{event.group_id}"
    return f"private_{event.user_id}"


# ── 群聊上下文收集（仅群消息） ──

group_listener = on_message(priority=1, block=False)


@group_listener.handle()
async def collect_group_context(bot: Bot, event: GroupMessageEvent) -> None:
    text = event.get_plaintext().strip()
    if not text:
        return
    nickname = event.sender.nickname or str(event.user_id)
    _group_ctx.add(
        group_id=str(event.group_id),
        user_id=str(event.user_id),
        nickname=nickname,
        content=text,
    )


# ── /identity 切换人设 ──

identity_cmd = on_command("identity", aliases={"人设"}, priority=5, block=True)


@identity_cmd.handle()
async def handle_identity(event: MessageEvent, args: Message = CommandArg()) -> None:  # noqa: B008
    arg = args.extract_plain_text().strip()
    sid = _session_id(event)

    if not arg or arg == "list":
        names = [f"{'* ' if i.id == 'default' else ''}{i.id} ({i.name})" for i in _identity_mgr.list_identities()]
        await identity_cmd.finish("可用人设:\n" + "\n".join(names))

    if arg == "reset":
        _identity_mgr.clear_override(sid)
        await identity_cmd.finish("已恢复自动匹配人设")

    result = _identity_mgr.switch(sid, arg)
    if result:
        await identity_cmd.finish(f"已切换人设: {result.name}")
    else:
        await identity_cmd.finish(f"未找到人设: {arg}")


# ── 对话 ──

chat = on_message(rule=to_me(), priority=10, block=True)


@chat.handle()
async def handle_chat(bot: Bot, event: MessageEvent) -> None:
    user_text = event.get_plaintext().strip()
    if not user_text:
        return

    sid = _session_id(event)
    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    identity = _identity_mgr.resolve(sid, group_id, user_text)

    ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=group_id)

    try:
        reply = await _llm.chat(
            session_id=sid,
            user_id=str(event.user_id),
            user_text=user_text,
            identity=identity,
            group_id=group_id,
            ctx=ctx,
        )
    except Exception:
        logger.exception("chat error")
        reply = "出错了，请稍后再试"

    await chat.finish(reply)
