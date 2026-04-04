"""对话插件：@机器人 触发，Soul + 记忆 + 工具 + 群聊上下文 + 主动插话。"""

import asyncio
import time
from datetime import timedelta

import aiohttp
from loguru import logger
from nonebot import get_driver, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageEvent
from nonebot.rule import to_me

from src.config import GroupConfig
from src.config_loader import load_config
from src.constants.qq_face import face_to_text
from src.identity import IdentityManager
from src.llm.client import (
    _RATE_LIMIT_BASE_DELAY,
    _RATE_LIMIT_MAX_RETRIES,
    LLMClient,
    RateLimitError,
)
from src.llm.dream import DreamAgent, setup_dream_logger
from src.llm.prompt import PromptBuilder, load_instruction
from src.llm.scheduler import GroupChatScheduler
from src.llm.usage import UsageTracker
from src.memory.group_timeline import GroupTimeline
from src.memory.history_loader import load_group_history
from src.memory.image_cache import ImageCache
from src.memory.memo_store import MemoStore
from src.memory.short_term import ShortTermMemory
from src.memory.types import Content, ContentBlock, ImageRefBlock, TextBlock
from src.sticker.store import StickerStore
from src.tools import ToolRegistry
from src.tools.context import ToolContext
from src.tools.datetime_tool import DateTimeTool
from src.tools.group_admin import MuteUserTool, SendGroupMsgTool, SetTitleTool
from src.tools.http_api import HttpApiTool
from src.tools.memo_tools import RecallMemoTool, UpdateMemoTool
from src.tools.sticker_tools import ManageStickerTool, SaveStickerTool, SendStickerTool
from src.tools.web_fetch import WebFetchTool
from src.tools.web_search import WebSearchTool

driver = get_driver()

_llm: LLMClient
_dream: DreamAgent
_dream_enabled: bool = False
_scheduler: GroupChatScheduler
_usage_tracker: UsageTracker
_identity_mgr: IdentityManager
_timeline: GroupTimeline
_short_term: ShortTermMemory
_allowed_groups: set[int] = set()
_group_config: GroupConfig = GroupConfig()
_allowed_private_users: set[int] = set()
_image_cache: ImageCache
_vision_enabled: bool = True
_max_images_per_message: int = 5
_sticker_store: StickerStore | None = None


@driver.on_startup
async def _init() -> None:
    global _llm, _dream, _dream_enabled, _scheduler, _identity_mgr, _usage_tracker
    global _timeline, _short_term, _allowed_groups, _allowed_private_users, _group_config
    global _image_cache, _vision_enabled, _max_images_per_message, _sticker_store

    bot_config = load_config()
    _allowed_groups = set(bot_config.group.allowed_groups)
    _group_config = bot_config.group
    _allowed_private_users = set(bot_config.allowed_private_users)

    _image_cache = ImageCache(
        cache_dir=bot_config.vision.cache_dir,
        max_dimension=bot_config.vision.max_dimension,
    )
    _vision_enabled = bot_config.vision.enabled
    _max_images_per_message = bot_config.vision.max_images_per_message

    # Cleanup stale cache on startup
    await _image_cache.cleanup(max_age=timedelta(hours=bot_config.vision.cache_max_age_hours))

    if bot_config.sticker.enabled:
        _sticker_store = StickerStore(
            storage_dir=bot_config.sticker.storage_dir,
            max_count=bot_config.sticker.max_count,
        )

    memo_store = MemoStore(
        base_dir=bot_config.memo.dir,
        history_enabled=bot_config.memo.history_enabled,
        user_max_chars=bot_config.memo.user_max_chars,
        group_max_chars=bot_config.memo.group_max_chars,
        index_max_lines=bot_config.memo.index_max_lines,
    )
    await memo_store.startup()
    _short_term = ShortTermMemory()
    _timeline = GroupTimeline()
    instruction = load_instruction(bot_config.soul.dir)
    short_term = _short_term

    superusers = set(bot_config.admins.keys()) | driver.config.superusers

    tools = ToolRegistry()
    tools.register(RecallMemoTool(memo_store))
    tools.register(UpdateMemoTool(memo_store))
    tools.register(DateTimeTool())
    tools.register(WebFetchTool())
    tools.register(WebSearchTool())
    tools.register(HttpApiTool())
    tools.register(MuteUserTool(superusers))
    tools.register(SetTitleTool(superusers))
    tools.register(SendGroupMsgTool(superusers))
    if _sticker_store is not None:
        tools.register(SaveStickerTool(_sticker_store, superusers))
        tools.register(SendStickerTool(_sticker_store))
        tools.register(ManageStickerTool(_sticker_store, superusers))

    _identity_mgr = IdentityManager()
    soul_dir = bot_config.soul.dir
    await _identity_mgr.load_file(f"{soul_dir}/identity.md")

    identity = _identity_mgr.resolve()
    prompt_builder = PromptBuilder(
        instruction=instruction,
        admins=bot_config.admins,
        sticker_store=_sticker_store,
    )
    prompt_builder.build_static(identity, bot_self_id="")

    _dream_enabled = bot_config.dream.enabled
    if _dream_enabled:
        setup_dream_logger(bot_config.log.dir)
    _dream = DreamAgent(
        store=memo_store,
        interval_hours=bot_config.dream.interval_hours,
        max_rounds=bot_config.dream.max_rounds,
        user_max_chars=bot_config.memo.user_max_chars,
        group_max_chars=bot_config.memo.group_max_chars,
        sticker_store=_sticker_store,
    )

    _usage_tracker = UsageTracker(db_path="storage/usage.db")
    if bot_config.llm.usage.enabled:
        await _usage_tracker.init()

    _llm = LLMClient(
        base_url=bot_config.llm.base_url,
        api_key=bot_config.llm.api_key,
        model=bot_config.llm.model,
        prompt_builder=prompt_builder,
        short_term=short_term,
        tools=tools,
        max_context_tokens=bot_config.llm.context.max_context_tokens,
        compact_ratio=bot_config.compact.ratio,
        compress_ratio=bot_config.compact.compress_ratio,
        max_compact_failures=bot_config.compact.max_failures,
        group_timeline=_timeline,
        memo_store=memo_store,
        on_compact=None,
        image_cache=_image_cache if _vision_enabled else None,
    )
    if bot_config.llm.usage.enabled:
        _llm._usage_tracker = _usage_tracker

    if bot_config.llm.usage.enabled:
        import nonebot

        from src.llm.usage_routes import create_usage_router

        app = nonebot.get_app()
        app.include_router(create_usage_router(_usage_tracker))

    _scheduler = GroupChatScheduler(
        llm=_llm,
        timeline=_timeline,
        identity_mgr=_identity_mgr,
        group_config=bot_config.group,
    )


@driver.on_shutdown
async def _shutdown() -> None:
    if _dream_enabled:
        await _dream.stop()
    await _llm.close()
    await _scheduler.close()
    await _usage_tracker.close()


@driver.on_bot_connect
async def _on_connect(bot: Bot) -> None:
    """Bot 连接后拉取群历史消息，填充群聊上下文。"""
    _llm._bot_self_id = bot.self_id
    # Rebuild static block now that we have the real bot_self_id
    _llm._prompt.build_static(_identity_mgr.resolve(), bot_self_id=bot.self_id)
    _scheduler.set_bot(bot)

    # Wire usage alert: PM all admins
    bot_config = load_config()
    admin_ids = list(bot_config.admins.keys())
    if admin_ids and bot_config.llm.usage.enabled:
        async def _alert_admins(msg: str) -> None:
            for admin_id in admin_ids:
                try:
                    await bot.send_private_msg(user_id=int(admin_id), message=msg)
                except Exception:
                    logger.warning("failed to send usage alert to admin {}", admin_id)

        _usage_tracker.set_alert(
            alert_fn=_alert_admins,
            cache_hit_warn=bot_config.compact.cache_hit_warn,
            slow_threshold_s=bot_config.llm.usage.slow_threshold_s,
            cache_alert_window_m=bot_config.compact.cache_alert_window_m,
            cache_alert_cooldown_m=bot_config.compact.cache_alert_cooldown_m,
        )

    try:
        group_list: list[dict[str, object]] = await bot.get_group_list()
        group_ids = [str(g["group_id"]) for g in group_list]
        if _allowed_groups:
            group_ids = [gid for gid in group_ids if int(gid) in _allowed_groups]
        logger.info("loading history | groups={}", len(group_ids))
        counts = {gid: _group_config.resolve(int(gid)).history_load_count for gid in group_ids}
        await load_group_history(
            napcat_url=bot_config.napcat.api_url,
            group_ids=group_ids,
            timeline=_timeline,
            count=bot_config.group.history_load_count,
            bot_self_id=bot.self_id,
            image_cache=_image_cache if _vision_enabled else None,
            sticker_store=_sticker_store,
            counts=counts,
        )
    except Exception:
        logger.exception("failed to load group history")
        return
    if _dream_enabled:
        _dream.start(_llm._call)

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
_REPLY_PREVIEW_MAX_SELF = 200  # longer preview when replying to bot's own message


async def _render_message(
    msg: Message,
    reply: object | None = None,
    session: aiohttp.ClientSession | None = None,
    self_id: str = "",
) -> Content:
    """将消息段转为文本或内容块列表。

    Returns plain str if no images; list[ContentBlock] if images present.
    """
    text_parts: list[str] = []
    images: list[ImageRefBlock] = []
    image_count = 0

    # 引用回复 → «回复 昵称(QQ号): 原文摘要»
    if reply is not None:
        sender = getattr(reply, "sender", None)
        reply_msg = getattr(reply, "message", None)
        if sender and reply_msg:
            uid = str(getattr(sender, "user_id", "") or "")
            nick = getattr(sender, "nickname", "") or uid
            is_reply_to_bot = self_id and uid == self_id
            cap = _REPLY_PREVIEW_MAX_SELF if is_reply_to_bot else _REPLY_PREVIEW_MAX
            original = reply_msg.extract_plain_text().strip()
            if len(original) > cap:
                original = original[:cap] + "…"
            label = "回复 我" if is_reply_to_bot else f"回复 {nick}({uid})"
            text_parts.append(f"«{label}: {original}» ")

    image_tasks: list[asyncio.Task[ImageRefBlock | None]] = []

    for seg in msg:
        if seg.type == "text":
            text_parts.append(seg.data.get("text", ""))
        elif seg.type == "at":
            qq = seg.data.get("qq", "")
            text_parts.append("@我" if self_id and qq == self_id else f"@{qq}")
        elif seg.type == "face":
            face_id = seg.data.get("id", "")
            try:
                text_parts.append(face_to_text(int(face_id)))
            except (ValueError, TypeError):
                text_parts.append("«表情»")
        elif seg.type == "image" and _vision_enabled and session is not None:
            if image_count >= _max_images_per_message:
                text_parts.append("«图片»")
                continue
            url = seg.data.get("url", "")
            file_id = seg.data.get("file", "")
            if url and file_id:
                file_id = file_id.split(".")[0] if "." in file_id else file_id
                image_tasks.append(
                    asyncio.ensure_future(_image_cache.save(session, url=url, file_id=file_id))
                )
                image_count += 1
            else:
                text_parts.append("«图片»")
        elif seg.type == "image":
            summary = seg.data.get("summary", "").strip("[]") or "图片"
            text_parts.append(f"«{summary}»")

    # Resolve all image downloads concurrently
    if image_tasks:
        t0 = time.perf_counter()
        results = await asyncio.gather(*image_tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException) or r is None:
                text_parts.append("«图片»")
            else:
                images.append(r)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "render_message images | tasks={} ok={} elapsed={:.0f}ms",
            len(image_tasks), len(images), elapsed_ms,
        )

    text = "".join(text_parts).strip()

    if not images:
        return text

    blocks: list[ContentBlock] = []
    if text:
        blocks.append(TextBlock(type="text", text=text))
    blocks.extend(images)
    return blocks


# ── 群聊上下文收集（仅群消息） ──

group_listener = on_message(priority=1, block=False)


@group_listener.handle()
async def collect_group_context(bot: Bot, event: GroupMessageEvent) -> None:
    if _allowed_groups and event.group_id not in _allowed_groups:
        return
    # Skip bot's own messages — already added as role="assistant" by LLMClient
    if str(event.user_id) == bot.self_id:
        return
    # Check per-group blocked users
    resolved = _group_config.resolve(event.group_id)
    if event.user_id in resolved.blocked_users:
        return
    content = await _render_message(event.get_message(), reply=event.reply, session=_llm._session, self_id=bot.self_id)
    if not content:
        return

    nickname = event.sender.nickname or str(event.user_id)
    group_id = str(event.group_id)
    preview = content if isinstance(content, str) else "".join(
        b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"  # type: ignore[union-attr]
    )
    if len(preview) > 120:
        preview = preview[:120] + "…"
    logger.info("group={} {}({}) | {}", group_id, nickname, event.user_id, preview)
    _timeline.add(
        group_id,
        role="user",
        speaker=f"{nickname}({event.user_id})",
        content=content,
        message_id=event.message_id,
    )

    _scheduler.notify(group_id, is_at=event.is_tome())


# ── 私聊 ──

private_chat = on_message(rule=to_me(), priority=10, block=True)


@private_chat.handle()
async def handle_private_chat(bot: Bot, event: MessageEvent) -> None:
    if isinstance(event, GroupMessageEvent):
        return
    if _allowed_private_users and event.user_id not in _allowed_private_users:
        return

    reply_msg = getattr(event, "reply", None)
    user_content = await _render_message(
        event.get_message(), reply=reply_msg, session=_llm._session, self_id=bot.self_id,
    )
    if not user_content:
        return

    sid = _session_id(event)
    identity = _identity_mgr.resolve()
    ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=None, session_id=sid)

    async def send_segment(text: str) -> None:
        await bot.send(event, Message(text))

    reply: str | None = None
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            reply = await _llm.chat(
                session_id=sid,
                user_id=str(event.user_id),
                user_content=user_content,
                identity=identity,
                group_id=None,
                ctx=ctx,
                on_segment=send_segment,
            )
            break
        except RateLimitError:
            if attempt >= _RATE_LIMIT_MAX_RETRIES:
                logger.error("private chat rate limit exhausted after {} retries", _RATE_LIMIT_MAX_RETRIES)
                reply = "当前请求太多，请稍后再试"
                break
            delay = _RATE_LIMIT_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "private chat rate limited, retry {}/{} in {:.0f}s",
                attempt + 1, _RATE_LIMIT_MAX_RETRIES, delay,
            )
            await asyncio.sleep(delay)
        except Exception:
            logger.exception("chat error")
            reply = "出错了，请稍后再试"
            break

    if reply:
        await private_chat.finish(Message(reply))
