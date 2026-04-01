"""对话插件：@机器人 触发，Soul + 记忆 + 工具 + 群聊上下文 + 主动插话。"""

from loguru import logger
from nonebot import get_driver, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, MessageEvent
from nonebot.rule import to_me

from src.config_loader import load_config
from src.identity import IdentityManager
from src.llm.client import LLMClient
from src.llm.proactive import ProactiveDecision, ProactiveEvaluator
from src.llm.prompt import PromptBuilder, load_instruction
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
_proactive: ProactiveEvaluator
_identity_mgr: IdentityManager
_timeline: GroupTimeline
_short_term: ShortTermMemory
_allowed_groups: set[int] = set()
_allowed_private_users: set[int] = set()


@driver.on_startup
async def _init() -> None:
    global _llm, _proactive, _identity_mgr, _timeline, _short_term, _allowed_groups, _allowed_private_users

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
        warm_enabled=bot_config.llm.cache.warm_enabled,
        warm_interval_messages=bot_config.llm.cache.warm_interval_messages,
        warm_ttl_seconds=bot_config.llm.cache.warm_ttl_seconds,
    )

    _proactive = ProactiveEvaluator(
        timeline=_timeline,
        model=bot_config.proactive.model,
        api_key=bot_config.llm.api_key,
        base_url=bot_config.llm.base_url,
        enabled=bot_config.proactive.enabled,
        timeout=bot_config.proactive.timeout,
        context_lines=bot_config.proactive.context_lines,
        cooldown=bot_config.proactive.cooldown,
        batch_timeout=bot_config.proactive.batch_timeout,
        batch_size=bot_config.proactive.batch_size,
    )


@driver.on_shutdown
async def _shutdown() -> None:
    await _llm.close()
    await _proactive.close()


@driver.on_bot_connect
async def _on_connect(bot: Bot) -> None:
    """Bot 连接后拉取群历史消息，填充群聊上下文。"""
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
        )
    except Exception:
        logger.exception("failed to load group history")
        return
    logger.info("Bot 就绪，开始接收消息 ✓")

    # 对历史消息触发一次主动插话检测，避免更新期间遗漏回复
    identity = _identity_mgr.resolve()
    for gid in group_ids:
        if not _timeline.get_messages(gid):
            continue

        async def _make_reply_callback(g: str) -> None:
            """为指定群构造主动回复回调。"""

            async def _on_reply(decision: ProactiveDecision) -> None:
                hint_parts: list[str] = []
                if decision["reply_to"]:
                    hint_parts.append(f"回复对象：{decision['reply_to']}")
                if decision["reason"]:
                    hint_parts.append(f"插话原因：{decision['reason']}")
                proactive_hint = "（主动插话）" + "，".join(hint_parts) if hint_parts else "（主动插话）"

                sid = f"group_{g}"
                ctx = ToolContext(bot=bot, user_id="", group_id=g)

                async def send_segment(seg_text: str) -> None:
                    await bot.send_group_msg(group_id=int(g), message=seg_text)

                reply = await _llm.chat(
                    session_id=sid,
                    user_id="",
                    user_text=f"[{proactive_hint}]",
                    identity=identity,
                    group_id=g,
                    ctx=ctx,
                    on_segment=send_segment,
                )
                if reply:
                    await bot.send_group_msg(group_id=int(g), message=reply)

            _proactive.notify(g, identity, _on_reply)

        await _make_reply_callback(gid)


def _session_id(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"group_{event.group_id}"
    return f"private_{event.user_id}"


# ── 群聊上下文收集（仅群消息） ──

group_listener = on_message(priority=1, block=False)


@group_listener.handle()
async def collect_group_context(bot: Bot, event: GroupMessageEvent) -> None:
    if _allowed_groups and event.group_id not in _allowed_groups:
        return
    text = event.get_plaintext().strip()
    if not text:
        return

    # 被 @ 的消息交给 chat handler，不走主动插话
    if event.is_tome():
        return

    nickname = event.sender.nickname or str(event.user_id)
    group_id = str(event.group_id)
    _timeline.add(
        group_id,
        role="user",
        speaker=f"{nickname}({event.user_id})",
        content=text,
    )

    identity = _identity_mgr.resolve()
    _llm.maybe_warm(group_id, identity, str(event.user_id))

    # 主动插话：通知 evaluator，由其 debounce/batch 后决定是否评估
    async def _on_proactive_reply(decision: ProactiveDecision) -> None:
        hint_parts: list[str] = []
        if decision["reply_to"]:
            hint_parts.append(f"回复对象：{decision['reply_to']}")
        if decision["reason"]:
            hint_parts.append(f"插话原因：{decision['reason']}")
        proactive_hint = "（主动插话）" + "，".join(hint_parts) if hint_parts else "（主动插话）"

        sid = _session_id(event)
        ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=group_id)

        async def send_segment(seg_text: str) -> None:
            await bot.send_group_msg(group_id=event.group_id, message=seg_text)

        reply = await _llm.chat(
            session_id=sid,
            user_id=str(event.user_id),
            user_text=f"[{proactive_hint}] {text}",
            identity=identity,
            group_id=group_id,
            ctx=ctx,
            on_segment=send_segment,
        )
        if reply:
            await bot.send_group_msg(group_id=event.group_id, message=reply)

    _proactive.notify(group_id, identity, _on_proactive_reply)


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

    user_text = event.get_plaintext().strip()
    if not user_text:
        return

    sid = _session_id(event)
    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    identity = _identity_mgr.resolve()

    ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=group_id)

    async def send_segment(text: str) -> None:
        await bot.send(event, text)

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
        await chat.finish(reply)
