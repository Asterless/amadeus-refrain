"""对话插件：@机器人 触发，Soul + 记忆 + 工具 + 群聊上下文 + 主动插话。"""

from loguru import logger
from nonebot import get_driver, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageEvent
from nonebot.rule import to_me

from src.config_loader import load_config
from src.identity import IdentityManager
from src.llm.client import LLMClient
from src.llm.prompt import PromptBuilder, load_instruction
from src.llm.scheduler import GroupChatScheduler
from src.memory.group_timeline import GroupTimeline
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

_llm: LLMClient
_scheduler: GroupChatScheduler
_identity_mgr: IdentityManager
_timeline: GroupTimeline
_short_term: ShortTermMemory
_allowed_groups: set[int] = set()
_allowed_private_users: set[int] = set()


@driver.on_startup
async def _init() -> None:
    global _llm, _scheduler, _identity_mgr, _timeline, _short_term, _allowed_groups, _allowed_private_users

    bot_config = load_config()
    _allowed_groups = set(bot_config.group.allowed_groups)
    _allowed_private_users = set(bot_config.allowed_private_users)

    long_term = LongTermMemory(memory_dir=bot_config.memory.dir)
    _short_term = ShortTermMemory()
    _timeline = GroupTimeline(max_messages=bot_config.group.max_timeline_messages)
    instruction = load_instruction(bot_config.soul.dir)
    prompt_builder = PromptBuilder(long_term=long_term, instruction=instruction)
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
    soul_dir = bot_config.soul.dir
    await _identity_mgr.load_file(f"{soul_dir}/identity.md")

    _llm = LLMClient(
        base_url=bot_config.llm.base_url,
        api_key=bot_config.llm.api_key,
        model=bot_config.llm.model,
        prompt_builder=prompt_builder,
        short_term=short_term,
        tools=tools,
        max_context_tokens=bot_config.llm.context.max_context_tokens,
        compact_ratio=bot_config.llm.context.compact_ratio,
        group_timeline=_timeline,
        long_term=long_term,
    )

    _scheduler = GroupChatScheduler(
        llm=_llm,
        timeline=_timeline,
        identity_mgr=_identity_mgr,
        debounce_seconds=bot_config.group.debounce_seconds,
        batch_size=bot_config.group.batch_size,
    )


@driver.on_shutdown
async def _shutdown() -> None:
    await _llm.close()
    await _scheduler.close()


@driver.on_bot_connect
async def _on_connect(bot: Bot) -> None:
    """Bot 连接后拉取群历史消息，填充群聊上下文。"""
    _llm._bot_self_id = bot.self_id
    _scheduler.set_bot(bot)
    try:
        bot_config = load_config()
        group_list: list[dict[str, object]] = await bot.get_group_list()
        group_ids = [str(g["group_id"]) for g in group_list]
        if _allowed_groups:
            group_ids = [gid for gid in group_ids if int(gid) in _allowed_groups]
        logger.info("loading history | groups={}", len(group_ids))
        await load_group_history(
            napcat_url=bot_config.napcat.api_url,
            group_ids=group_ids,
            timeline=_timeline,
            count=bot_config.group.history_load_count,
            bot_self_id=bot.self_id,
        )
    except Exception:
        logger.exception("failed to load group history")
        return
    logger.info("Bot 就绪，开始接收消息 ✓")


def _session_id(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"group_{event.group_id}"
    return f"private_{event.user_id}"


def _render_message(msg: Message) -> str:
    """将消息段转为可读文本，保留 @提及 信息。"""
    parts: list[str] = []
    for seg in msg:
        if seg.type == "text":
            parts.append(seg.data.get("text", ""))
        elif seg.type == "at":
            qq = seg.data.get("qq", "")
            parts.append(f"@{qq}")
    return "".join(parts).strip()


# ── 群聊上下文收集（仅群消息） ──

group_listener = on_message(priority=1, block=False)


@group_listener.handle()
async def collect_group_context(bot: Bot, event: GroupMessageEvent) -> None:
    if _allowed_groups and event.group_id not in _allowed_groups:
        return
    # Skip bot's own messages — already added as role="assistant" by LLMClient
    if str(event.user_id) == bot.self_id:
        return
    text = _render_message(event.get_message())
    if not text:
        return

    nickname = event.sender.nickname or str(event.user_id)
    group_id = str(event.group_id)
    _timeline.add(
        group_id,
        role="user",
        speaker=f"{nickname}({event.user_id})",
        content=text,
    )

    if event.is_tome():
        return

    _scheduler.notify(group_id)


# ── 对话 ──

chat = on_message(rule=to_me(), priority=10, block=True)


@chat.handle()
async def handle_chat(bot: Bot, event: MessageEvent) -> None:
    # 白名单过滤
    if isinstance(event, GroupMessageEvent):
        if _allowed_groups and event.group_id not in _allowed_groups:
            return
    else:
        if _allowed_private_users and event.user_id not in _allowed_private_users:
            return

    user_text = _render_message(event.get_message())
    if not user_text:
        return

    sid = _session_id(event)
    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    identity = _identity_mgr.resolve()

    if group_id:
        _scheduler.interrupt(group_id)

    ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=group_id)

    async def send_segment(text: str) -> None:
        await bot.send(event, Message(text))

    try:
        reply = await _llm.chat(
            session_id=sid,
            user_id=str(event.user_id),
            user_text=user_text,
            identity=identity,
            group_id=group_id,
            ctx=ctx,
            on_segment=send_segment,
        )
    except Exception:
        logger.exception("chat error")
        reply = "出错了，请稍后再试"

    if reply:
        await chat.finish(Message(reply))
