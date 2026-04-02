"""对话插件：@机器人 触发，Soul + 记忆 + 工具 + 群聊上下文 + 主动插话。"""

from loguru import logger
from nonebot import get_driver, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageEvent
from nonebot.rule import to_me

from src.config_loader import load_config
from src.identity import IdentityManager
from src.llm.client import LLMClient
from src.llm.dream import DreamAgent
from src.llm.prompt import PromptBuilder, load_instruction
from src.llm.scheduler import GroupChatScheduler
from src.memory.group_timeline import GroupTimeline
from src.memory.history_loader import load_group_history
from src.memory.memo_store import MemoStore
from src.memory.short_term import ShortTermMemory
from src.tools import ToolRegistry
from src.tools.context import ToolContext
from src.tools.datetime_tool import DateTimeTool
from src.tools.group_admin import MuteUserTool, SendGroupMsgTool, SetTitleTool
from src.tools.http_api import HttpApiTool
from src.tools.memo_tools import RecallMemoTool, UpdateMemoTool
from src.tools.web_fetch import WebFetchTool

driver = get_driver()

_llm: LLMClient
_dream: DreamAgent
_dream_enabled: bool = False
_scheduler: GroupChatScheduler
_identity_mgr: IdentityManager
_timeline: GroupTimeline
_short_term: ShortTermMemory
_allowed_groups: set[int] = set()
_allowed_private_users: set[int] = set()


@driver.on_startup
async def _init() -> None:
    global _llm, _dream, _dream_enabled, _scheduler, _identity_mgr
    global _timeline, _short_term, _allowed_groups, _allowed_private_users

    bot_config = load_config()
    _allowed_groups = set(bot_config.group.allowed_groups)
    _allowed_private_users = set(bot_config.allowed_private_users)

    memo_store = MemoStore(
        base_dir=bot_config.memo.dir,
        history_enabled=bot_config.memo.history_enabled,
        user_max_chars=bot_config.memo.user_max_chars,
        group_max_chars=bot_config.memo.group_max_chars,
        index_max_lines=bot_config.memo.index_max_lines,
    )
    await memo_store.startup()
    _short_term = ShortTermMemory()
    _timeline = GroupTimeline(max_messages=bot_config.group.max_timeline_messages)
    instruction = load_instruction(bot_config.soul.dir)
    short_term = _short_term

    superusers = bot_config.superusers | driver.config.superusers

    tools = ToolRegistry()
    tools.register(RecallMemoTool(memo_store))
    tools.register(UpdateMemoTool(memo_store))
    tools.register(DateTimeTool())
    tools.register(WebFetchTool())
    tools.register(HttpApiTool())
    tools.register(MuteUserTool(superusers))
    tools.register(SetTitleTool(superusers))
    tools.register(SendGroupMsgTool(superusers))

    _identity_mgr = IdentityManager()
    soul_dir = bot_config.soul.dir
    await _identity_mgr.load_file(f"{soul_dir}/identity.md")

    identity = _identity_mgr.resolve()
    prompt_builder = PromptBuilder(instruction=instruction, admins=bot_config.admins)
    prompt_builder.build_static(identity, bot_self_id="")

    _dream_enabled = bot_config.dream.enabled
    _dream = DreamAgent(
        store=memo_store,
        interval_hours=bot_config.dream.interval_hours,
        min_compacts=bot_config.dream.min_compacts,
        max_rounds=bot_config.dream.max_rounds,
        user_max_chars=bot_config.memo.user_max_chars,
        group_max_chars=bot_config.memo.group_max_chars,
    )

    _llm = LLMClient(
        base_url=bot_config.llm.base_url,
        api_key=bot_config.llm.api_key,
        model=bot_config.llm.model,
        prompt_builder=prompt_builder,
        short_term=short_term,
        tools=tools,
        max_context_tokens=bot_config.llm.context.max_context_tokens,
        micro_ratio=bot_config.compact.micro_ratio,
        full_ratio=bot_config.compact.full_ratio,
        max_compact_failures=bot_config.compact.max_failures,
        cache_hit_warn=bot_config.compact.cache_hit_warn,
        group_timeline=_timeline,
        memo_store=memo_store,
        on_compact=lambda: _dream.notify_compact(),
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
    # Rebuild static block now that we have the real bot_self_id
    _llm._prompt.build_static(_identity_mgr.resolve(), bot_self_id=bot.self_id)
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

    # Evaluate history for each group — catch up on missed messages
    for gid in group_ids:
        if _timeline.get_messages(gid):
            _scheduler.trigger(gid)


def _session_id(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"group_{event.group_id}"
    return f"private_{event.user_id}"


_REPLY_PREVIEW_MAX = 50


def _render_message(msg: Message, reply: object | None = None) -> str:
    """将消息段转为可读文本，保留 @提及 和引用回复信息。"""
    parts: list[str] = []

    # 引用回复 → [回复 昵称(QQ号): 原文摘要]
    if reply is not None:
        sender = getattr(reply, "sender", None)
        reply_msg = getattr(reply, "message", None)
        if sender and reply_msg:
            uid = getattr(sender, "user_id", "") or ""
            nick = getattr(sender, "nickname", "") or str(uid)
            original = reply_msg.extract_plain_text().strip()
            if len(original) > _REPLY_PREVIEW_MAX:
                original = original[:_REPLY_PREVIEW_MAX] + "…"
            parts.append(f"[回复 {nick}({uid}): {original}] ")

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
    text = _render_message(event.get_message(), reply=event.reply)
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


async def _dream_llm_call(system_prompt: str) -> None:
    """Placeholder Dream LLM call — will be fleshed out when Dream is fully integrated."""
    logger.warning("dream LLM call is a STUB — dream consolidation is NOT running (prompt len={})", len(system_prompt))


@chat.handle()
async def handle_chat(bot: Bot, event: MessageEvent) -> None:
    # 白名单过滤
    if isinstance(event, GroupMessageEvent):
        if _allowed_groups and event.group_id not in _allowed_groups:
            return
    else:
        if _allowed_private_users and event.user_id not in _allowed_private_users:
            return

    reply = getattr(event, "reply", None)
    user_text = _render_message(event.get_message(), reply=reply)
    if not user_text:
        return

    sid = _session_id(event)
    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    identity = _identity_mgr.resolve()

    if group_id:
        _scheduler.interrupt(group_id)

    ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=group_id, session_id=sid)

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
    finally:
        if group_id:
            _scheduler.release(group_id)

    # Check if Dream should run (fire-and-forget, gated by config)
    if _dream_enabled:
        await _dream.maybe_run(_dream_llm_call)

    if reply:
        await chat.finish(Message(reply))
