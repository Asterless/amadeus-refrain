"""对话插件：@机器人 触发，Soul + 记忆 + 工具 + 群聊上下文 + 主动插话。"""

import asyncio
import time
from datetime import timedelta
from pathlib import Path

import aiohttp
from loguru import logger
from nonebot import get_driver, on_message, on_notice
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupBanNoticeEvent,
    GroupMessageEvent,
    Message,
    MessageEvent,
)
from nonebot.rule import Rule, to_me

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
from src.meme import MemeKnowledgeStore, MemeRadar, MemeStore, UapiTrendProvider
from src.memory.group_timeline import GroupTimeline
from src.memory.history_loader import load_group_history
from src.memory.image_cache import ImageCache
from src.memory.memo_store import MemoStore
from src.memory.message_log import MessageLog
from src.memory.short_term import ShortTermMemory
from src.memory.types import Content, ContentBlock, ImageRefBlock, TextBlock
from src.music import NeteaseMusicClient
from src.sticker.store import StickerStore
from src.tools import ToolRegistry
from src.tools.context import ToolContext
from src.tools.datetime_tool import DateTimeTool
from src.tools.group_admin import MuteUserTool, SendGroupMsgTool, SetTitleTool
from src.tools.http_api import HttpApiTool
from src.tools.imagegen_tools import EditImageTool, GenerateImageTool
from src.tools.meme_learning import SaveMemeKnowledgeTool
from src.tools.meme_tools import GetHotTrendsTool, SearchMemeTool
from src.tools.memo_tools import RecallMemoTool, UpdateMemoTool
from src.tools.music_tools import (
    MusicLoginStatusTool,
    MusicQrLoginTool,
    MusicSearchTool,
    MusicShareTool,
)
from src.tools.sticker_tools import ManageStickerTool, SaveStickerTool, SendStickerTool, _deliver_sticker
from src.tools.voice_tools import SendVoiceTool
from src.tools.web_fetch import WebFetchTool
from src.tools.web_search import HybridWebSearch, OpenAIWebSearchClient, WebSearchTool
from src.vision import VisionClient

driver = get_driver()

_llm: LLMClient
_dream: DreamAgent
_dream_enabled: bool = False
_meme_radar: MemeRadar | None = None
_music_client: NeteaseMusicClient | None = None
_search_service: HybridWebSearch | None = None
_meme_knowledge: MemeKnowledgeStore | None = None
_scheduler: GroupChatScheduler
_usage_tracker: UsageTracker
_message_log: MessageLog
_identity_mgr: IdentityManager
_timeline: GroupTimeline
_short_term: ShortTermMemory
_allowed_groups: set[int] = set()
_group_config: GroupConfig = GroupConfig()
_allowed_private_users: set[int] = set()
_superusers: set[str] = set()
_image_cache: ImageCache
_vision_enabled: bool = True
_max_images_per_message: int = 5
_sticker_store: StickerStore | None = None
_sticker_sender: SendStickerTool | None = None
_voice_sender: SendVoiceTool | None = None
_imagegen_tool: GenerateImageTool | None = None
_imagegen_edit_tool: EditImageTool | None = None
_vision_client: VisionClient | None = None
_sticker_auto_collect: bool = True
_sticker_auto_collect_only_stickers: bool = True
_sticker_auto_collect_cooldown: int = 8
_sticker_collect_last: dict[str, float] = {}
_startup_triggered: bool = False
_restart_notice_sent: bool = False


def _imagegen_pending_rule(event: MessageEvent) -> bool:
    """仅当该用户/群存在未过期的生图确认请求时触发确认处理。"""
    if _imagegen_tool is None:
        return False
    text = event.get_message().extract_plain_text().strip()
    if text.startswith("/"):
        return False
    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    reply_message_id = event.reply.message_id if event.reply is not None else None
    return _imagegen_tool.can_handle_confirmation(
        str(event.user_id), group_id, text, reply_message_id,
    )


_imagegen_confirm = on_message(
    priority=0, block=True, rule=Rule(_imagegen_pending_rule),
)


@_imagegen_confirm.handle()
async def handle_imagegen_confirm(bot: Bot, event: MessageEvent) -> None:
    if _imagegen_tool is None:
        return
    group_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else None
    text = event.get_message().extract_plain_text().strip()
    reply_message_id = event.reply.message_id if event.reply is not None else None
    try:
        await _imagegen_tool.try_confirm(
            bot=bot, user_id=str(event.user_id), group_id=group_id, text=text,
            reply_message_id=reply_message_id,
        )
    except Exception:
        logger.warning("imagegen confirm handling failed", exc_info=True)


@driver.on_startup
async def _init() -> None:
    global _llm, _dream, _dream_enabled, _meme_radar, _music_client, _search_service, _meme_knowledge, _scheduler
    global _identity_mgr, _usage_tracker, _superusers
    global _message_log, _timeline, _short_term, _allowed_groups, _allowed_private_users, _group_config
    global _image_cache, _vision_enabled, _max_images_per_message, _sticker_store, _vision_client
    global _sticker_sender, _voice_sender, _imagegen_tool, _imagegen_edit_tool
    global _sticker_auto_collect, _sticker_auto_collect_only_stickers, _sticker_auto_collect_cooldown

    bot_config = load_config()
    logger.info("[Startup][Chat] initializing chat plugin")
    _allowed_groups = set(bot_config.group.allowed_groups)
    _group_config = bot_config.group
    _allowed_private_users = set(bot_config.allowed_private_users)

    _image_cache = ImageCache(
        cache_dir=bot_config.vision.cache_dir,
        max_dimension=bot_config.vision.max_dimension,
    )
    _vision_enabled = bot_config.vision.enabled
    _max_images_per_message = bot_config.vision.max_images_per_message
    _sticker_auto_collect = bot_config.sticker.auto_collect
    _sticker_auto_collect_only_stickers = bot_config.sticker.auto_collect_only_stickers
    _sticker_auto_collect_cooldown = bot_config.sticker.auto_collect_cooldown_seconds
    logger.info(
        "[Startup][Vision] enabled={} describe_mode={} max_images={}",
        _vision_enabled, bot_config.vision.describe_mode, _max_images_per_message,
    )

    # Cleanup stale cache on startup
    await _image_cache.cleanup(max_age=timedelta(hours=bot_config.vision.cache_max_age_hours))

    if bot_config.sticker.enabled:
        _sticker_store = StickerStore(
            storage_dir=bot_config.sticker.storage_dir,
            max_count=bot_config.sticker.max_count,
        )
        synced = _sticker_store.sync_from_disk()
        if synced["added"] or synced["skipped"] or synced["duplicates"]:
            logger.info(
                "sticker sync_from_disk | added={} skipped={} duplicates={}",
                synced["added"], synced["skipped"], synced["duplicates"],
            )
        logger.info(
            "[Startup][Sticker] enabled=true count={} max_count={} send_probability={:.0%} auto_collect={}",
            len(_sticker_store.list_all()), bot_config.sticker.max_count,
            bot_config.sticker.send_probability, _sticker_auto_collect,
        )
    else:
        logger.info("[Startup][Sticker] enabled=false")

    # 免费识图预处理：聊天模型不支持图片时，先用视觉模型把图转成文字描述
    _vision_client = None
    if (
        bot_config.vision.enabled
        and bot_config.vision.base_url
        and bot_config.vision.api_key
        and bot_config.vision.model
    ):
        _vision_client = VisionClient(
            base_url=bot_config.vision.base_url,
            api_key=bot_config.vision.api_key,
            model=bot_config.vision.model,
        )
        logger.info(
            "[Vision] 识图预处理已启用 | model={} base_url={} describe_mode={}",
            bot_config.vision.model,
            bot_config.vision.base_url,
            bot_config.vision.describe_mode,
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
    instruction = load_instruction(bot_config.soul.dir)
    short_term = _short_term

    superusers = set(bot_config.admins.keys()) | driver.config.superusers
    _superusers = superusers

    _music_client = None
    if bot_config.music.enabled:
        _music_client = NeteaseMusicClient(
            bot_config.music.api_base_url,
            cookie_file=bot_config.music.cookie_file,
            timeout=bot_config.music.timeout_seconds,
        )
        if bot_config.music.auto_start and bot_config.music.service_app:
            try:
                ready = await _music_client.start_local_service(
                    bot_config.music.service_app,
                    node_executable=bot_config.music.node_executable,
                )
                logger.info("[Music] local API auto-start | ready={}", ready)
            except (OSError, ValueError):
                logger.warning("[Music] local API auto-start failed", exc_info=True)
        logger.info(
            "[Startup][Music] enabled=true api={} auto_start={}",
            bot_config.music.api_base_url, bot_config.music.auto_start,
        )
    else:
        logger.info("[Startup][Music] enabled=false")

    meme_store: MemeStore | None = None
    if bot_config.meme.enabled:
        meme_store = MemeStore(
            bot_config.meme.storage_file,
            active_hours=bot_config.meme.active_hours,
            max_entries=bot_config.meme.max_entries,
            max_prompt_entries=bot_config.meme.max_prompt_entries,
        )
        _meme_knowledge = MemeKnowledgeStore(bot_config.meme.knowledge_file)
    else:
        _meme_knowledge = None

    _search_service = None
    if bot_config.search.openai_enabled:
        if bot_config.search.openai_api_key and bot_config.search.openai_model:
            _search_service = HybridWebSearch(
                OpenAIWebSearchClient(
                    base_url=bot_config.search.openai_base_url,
                    api_key=bot_config.search.openai_api_key,
                    model=bot_config.search.openai_model,
                )
            )
            logger.info(
                "[Search] OpenAI web_search enabled | model={} base_url={}",
                bot_config.search.openai_model,
                bot_config.search.openai_base_url,
            )
        else:
            logger.warning("[Search] OpenAI web_search requested but API key or model is empty")

    web_search = _search_service.search if _search_service is not None else None
    tools = ToolRegistry()
    tools.register(RecallMemoTool(memo_store))
    tools.register(UpdateMemoTool(memo_store))
    tools.register(DateTimeTool())
    tools.register(WebFetchTool())
    tools.register(WebSearchTool(web_search))
    if meme_store is not None:
        tools.register(GetHotTrendsTool(meme_store))
        tools.register(SearchMemeTool(meme_store, web_search, _meme_knowledge))
        if _meme_knowledge is not None:
            tools.register(SaveMemeKnowledgeTool(_meme_knowledge, on_change=lambda: prompt_builder.invalidate()))
    if _music_client is not None:
        tools.register(MusicSearchTool(_music_client))
        tools.register(MusicShareTool(_music_client))
        tools.register(MusicQrLoginTool(_music_client, superusers))
        tools.register(MusicLoginStatusTool(_music_client, superusers))
    if bot_config.tts.enabled:
        tools.register(
            SendVoiceTool(
                provider=bot_config.tts.provider,
                voice=bot_config.tts.voice,
                rate=bot_config.tts.rate,
                volume=bot_config.tts.volume,
                proxy=bot_config.tts.proxy,
                base_url=bot_config.tts.base_url,
                ref_audio_path=bot_config.tts.ref_audio_path,
                prompt_text=bot_config.tts.prompt_text,
                prompt_lang=bot_config.tts.prompt_lang,
                text_lang=bot_config.tts.text_lang,
                text_split_method=bot_config.tts.text_split_method,
                media_type=bot_config.tts.media_type,
                timeout_seconds=bot_config.tts.timeout_seconds,
                max_chars=bot_config.tts.max_chars,
            )
        )
        logger.info(
            "[Startup][TTS] enabled=true provider={} voice={} max_chars={}",
            bot_config.tts.provider, bot_config.tts.voice, bot_config.tts.max_chars,
        )
    else:
        logger.info("[Startup][TTS] enabled=false")
    if bot_config.imagegen.enabled:
        imagegen_api_key = bot_config.imagegen.api_key
        if not imagegen_api_key and "bigmodel.cn" in bot_config.imagegen.base_url:
            imagegen_api_key = bot_config.vision.api_key
        _imagegen_tool = GenerateImageTool(
            base_url=bot_config.imagegen.base_url,
            api_key=imagegen_api_key,
            model=bot_config.imagegen.model,
            size=bot_config.imagegen.size,
            timeout_seconds=bot_config.imagegen.timeout_seconds,
            max_prompt_chars=bot_config.imagegen.max_prompt_chars,
            proxy=bot_config.imagegen.proxy,
            daily_global_limit=bot_config.imagegen.daily_global_limit,
            daily_user_limit=bot_config.imagegen.daily_user_limit,
            daily_group_limit=bot_config.imagegen.daily_group_limit,
            cooldown_seconds=bot_config.imagegen.cooldown_seconds,
            usage_file=bot_config.imagegen.usage_file,
        )
        _imagegen_edit_tool = EditImageTool(
            base_url=bot_config.imagegen.base_url,
            api_key=imagegen_api_key,
            model=bot_config.imagegen.model,
            size=bot_config.imagegen.size,
            timeout_seconds=bot_config.imagegen.timeout_seconds,
            max_prompt_chars=bot_config.imagegen.max_prompt_chars,
            proxy=bot_config.imagegen.proxy,
            daily_global_limit=bot_config.imagegen.daily_global_limit,
            daily_user_limit=bot_config.imagegen.daily_user_limit,
            daily_group_limit=bot_config.imagegen.daily_group_limit,
            cooldown_seconds=bot_config.imagegen.cooldown_seconds,
            usage_file=bot_config.imagegen.usage_file,
            usage=_imagegen_tool.usage,
        )
        tools.register(
            _imagegen_tool
        )
        tools.register(_imagegen_edit_tool)
        logger.info(
            "[Startup][ImageGen] enabled=true model={} size={} proxy={} edit=True "
            "limits=global:{}/user:{}/group:{} cooldown={}s api_key_set={}",
            bot_config.imagegen.model,
            bot_config.imagegen.size,
            bot_config.imagegen.proxy or "none",
            bot_config.imagegen.daily_global_limit,
            bot_config.imagegen.daily_user_limit,
            bot_config.imagegen.daily_group_limit,
            bot_config.imagegen.cooldown_seconds,
            bool(imagegen_api_key),
        )
    else:
        logger.info("[Startup][ImageGen] enabled=false")
    tools.register(HttpApiTool())
    tools.register(MuteUserTool(superusers))
    tools.register(SetTitleTool(superusers))
    tools.register(SendGroupMsgTool(superusers))
    if _sticker_store is not None:
        tools.register(SaveStickerTool(_sticker_store, superusers))
        # The model must first choose a relevant sticker; config controls the final send gate.
        tools.register(SendStickerTool(_sticker_store, send_probability=bot_config.sticker.send_probability))
        tools.register(ManageStickerTool(_sticker_store, superusers))
    _sticker_sender = tools.get("send_sticker")  # type: ignore[assignment]
    _voice_sender = tools.get("send_voice")  # type: ignore[assignment]
    logger.info("[Startup][Tools] registered count={} names={}", len(tools.names), ",".join(tools.names))

    _identity_mgr = IdentityManager()
    soul_dir = bot_config.soul.dir
    await _identity_mgr.load_file(f"{soul_dir}/identity.md")

    identity = _identity_mgr.resolve()
    prompt_builder = PromptBuilder(
        instruction=instruction,
        admins={**bot_config.admins, **{uid: "管理员" for uid in superusers if uid not in bot_config.admins}},
        sticker_store=_sticker_store,
        meme_store=meme_store,
        meme_knowledge=_meme_knowledge,
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
        on_memo_change=lambda: prompt_builder.invalidate(),
    )
    logger.info("[Startup][Dream] enabled={} interval={}h", _dream_enabled, bot_config.dream.interval_hours)

    _usage_tracker = UsageTracker(db_path="storage/usage.db")
    if bot_config.llm.usage.enabled:
        await _usage_tracker.init()
    logger.info("[Startup][Usage] enabled={}", bot_config.llm.usage.enabled)

    _message_log = MessageLog(db_path="storage/messages.db")
    await _message_log.init()

    _timeline = GroupTimeline(message_log=_message_log)

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
        vision_client=_vision_client,
        message_log=_message_log,
    )
    rewrite_model = bot_config.imagegen.prompt_rewrite_model

    async def rewrite_image_prompt(current_prompt: str, revision: str) -> str:
        return await _llm.rewrite_image_prompt(current_prompt, revision, model=rewrite_model)

    if _imagegen_tool is not None:
        _imagegen_tool.set_prompt_rewriter(rewrite_image_prompt)
    if _imagegen_edit_tool is not None:
        _imagegen_edit_tool.set_prompt_rewriter(rewrite_image_prompt)
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
        always_describe_images=bot_config.vision.describe_mode == "always",
        reply_on_sticker=bot_config.sticker.enabled and bot_config.sticker.reply_on_receive,
        auto_sticker_sender=_sticker_sender,
    )
    _meme_radar = None
    if meme_store is not None:
        provider = UapiTrendProvider(bot_config.meme.hotboard_url)
        _meme_radar = MemeRadar(
            meme_store,
            provider,
            platforms=bot_config.meme.platforms,
            refresh_minutes=bot_config.meme.refresh_minutes,
            per_platform_limit=bot_config.meme.per_platform_limit,
            on_change=prompt_builder.invalidate,
        )
        _meme_radar.start()
        logger.info("[Startup][MemeRadar] started refresh={}min", bot_config.meme.refresh_minutes)

    logger.info("[Startup][Chat] initialization complete")


@driver.on_shutdown
async def _shutdown() -> None:
    if _dream_enabled:
        await _dream.stop()
    if _meme_radar is not None:
        await _meme_radar.stop()
    if _search_service is not None:
        await _search_service.close()
    if _meme_knowledge is not None:
        _meme_knowledge.close()
    if _music_client is not None:
        await _music_client.close()
    await _llm.close()
    await _scheduler.close()
    await _message_log.close()
    if _vision_client is not None:
        await _vision_client.close()
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

    global _startup_triggered
    is_first_connect = not _startup_triggered

    try:
        group_list: list[dict[str, object]] = await bot.get_group_list()
        group_ids = [str(g["group_id"]) for g in group_list]
        if _allowed_groups:
            group_ids = [gid for gid in group_ids if int(gid) in _allowed_groups]
    except Exception:
        logger.exception("failed to get group list")
        return

    if is_first_connect and bot_config.group.startup_catchup:
        _startup_triggered = True
        logger.info("loading history | groups={}", len(group_ids))
        try:
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
    else:
        logger.info("reconnected, skipping history reload (already loaded)")

    # Check bot mute status in each group
    muted_count = 0
    for gid in group_ids:
        try:
            info: dict[str, object] = await bot.get_group_member_info(
                group_id=int(gid), user_id=int(bot.self_id),
            )
            raw = info.get("shut_up_timestamp") or 0
            shut_until = int(str(raw))
            if shut_until > time.time():
                _scheduler.mute(gid)
                muted_count += 1
        except Exception:
            logger.debug("failed to query mute status | group={}", gid)
    if muted_count:
        logger.info("muted in {} group(s) at startup", muted_count)

    logger.info("Bot 就绪，开始接收消息 ✓")

    global _restart_notice_sent
    if not _restart_notice_sent:
        _restart_notice_sent = True
        await _send_restart_notice(bot)

    # Evaluate history for each group — catch up on missed messages (first connect only)
    if is_first_connect:
        for gid in group_ids:
            if _timeline.get_turns(gid) or _timeline.get_pending(gid):
                _scheduler.trigger(gid)


async def _send_restart_notice(bot: Bot) -> None:
    """进程启动/重启完成后仅私聊通知管理员。"""
    message = "重启完成，我回来了 ✓"
    bot_config = load_config()
    for admin_id in bot_config.admins:
        try:
            await bot.send_private_msg(user_id=int(admin_id), message=message)
        except Exception:
            logger.warning("failed to send restart notice to admin {}", admin_id)
    logger.info("restart notice sent | admins={}", len(bot_config.admins))


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
    ordered_parts: list[str | ImageRefBlock | asyncio.Task[ImageRefBlock | None]] = []
    image_count = 0
    image_tasks: list[asyncio.Task[ImageRefBlock | None]] = []

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
            ordered_parts.append(f"«{label}: {original}» ")
            for reply_seg in reply_msg:
                if reply_seg.type != "image" or session is None:
                    continue
                if image_count >= _max_images_per_message:
                    break
                url = reply_seg.data.get("url", "")
                file_id = reply_seg.data.get("file", "")
                if not url or not file_id:
                    continue
                file_id = file_id.split(".")[0] if "." in file_id else file_id
                is_sticker_segment = str(
                    reply_seg.data.get("subType", reply_seg.data.get("sub_type", "0"))
                ) == "1"
                task = asyncio.ensure_future(
                    _image_cache.save(
                        session, url=url, file_id=file_id, preserve_original=is_sticker_segment,
                    )
                )
                image_tasks.append(task)
                ordered_parts.append(task)
                image_count += 1

    for seg in msg:
        if seg.type == "text":
            ordered_parts.append(seg.data.get("text", ""))
        elif seg.type == "at":
            qq = seg.data.get("qq", "")
            if self_id and qq == self_id:
                ordered_parts.append("@我")
            else:
                name = seg.data.get("name", "") or ""
                ordered_parts.append(f"@{name}({qq})" if name else f"@{qq}")
        elif seg.type == "face":
            face_id = seg.data.get("id", "")
            try:
                ordered_parts.append(face_to_text(int(face_id)))
            except (ValueError, TypeError):
                ordered_parts.append("«表情»")
        elif seg.type == "image" and session is not None and (
            _vision_enabled
            or str(seg.data.get("subType", seg.data.get("sub_type", "0"))) == "1"
        ):
            if image_count >= _max_images_per_message:
                ordered_parts.append("«图片»")
                continue
            url = seg.data.get("url", "")
            file_id = seg.data.get("file", "")
            if url and file_id:
                file_id = file_id.split(".")[0] if "." in file_id else file_id
                is_sticker_segment = str(seg.data.get("subType", seg.data.get("sub_type", "0"))) == "1"
                task = asyncio.ensure_future(
                    _image_cache.save(
                        session, url=url, file_id=file_id, preserve_original=is_sticker_segment,
                    )
                )
                image_tasks.append(task)
                ordered_parts.append(task)
                image_count += 1
            else:
                ordered_parts.append("«图片»")
        elif seg.type == "image":
            summary = seg.data.get("summary", "").strip("[]") or "图片"
            ordered_parts.append(f"«{summary}»")

    # Resolve all image downloads concurrently
    if image_tasks:
        t0 = time.perf_counter()
        results = await asyncio.gather(*image_tasks, return_exceptions=True)
        result_by_task = dict(zip(image_tasks, results, strict=True))
        resolved_parts: list[str | ImageRefBlock] = []
        for part in ordered_parts:
            if isinstance(part, asyncio.Task):
                result = result_by_task[part]
                resolved_parts.append("«图片»" if isinstance(result, BaseException) or result is None else result)
            else:
                resolved_parts.append(part)
        final_parts = resolved_parts
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "render_message images | tasks={} ok={} elapsed={:.0f}ms",
            len(image_tasks), sum(isinstance(p, dict) for p in final_parts), elapsed_ms,
        )
    else:
        final_parts = [part for part in ordered_parts if not isinstance(part, asyncio.Task)]

    if not any(isinstance(part, dict) for part in final_parts):
        return "".join(part for part in final_parts if isinstance(part, str)).strip()

    blocks: list[ContentBlock] = []
    text_buffer: list[str] = []
    for part in final_parts:
        if isinstance(part, str):
            text_buffer.append(part)
            continue
        text = "".join(text_buffer).strip()
        if text:
            blocks.append(TextBlock(type="text", text=text))
        text_buffer.clear()
        blocks.append(part)
    text = "".join(text_buffer).strip()
    if text:
        blocks.append(TextBlock(type="text", text=text))
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
    # Muted — pause listening entirely
    if _scheduler.is_muted(str(event.group_id)):
        return
    # Check per-group blocked users
    resolved = _group_config.resolve(event.group_id)
    if event.user_id in resolved.blocked_users:
        return
    content = await _render_message(event.get_message(), reply=event.reply, session=_llm._session, self_id=bot.self_id)
    if not content:
        return

    # 群聊优先用群名片（card），全局昵称（nickname）与群里看到的称呼可能不一致
    nickname = event.sender.card or event.sender.nickname or str(event.user_id)
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

    is_sticker = any(
        seg.type == "image"
        and str(seg.data.get("subType", seg.data.get("sub_type", "0"))) == "1"
        for seg in event.get_message()
    )
    await _auto_collect_stickers(bot, content, is_sticker, str(event.group_id))
    reply_sender = getattr(event.reply, "sender", None) if event.reply is not None else None
    reply_user_id = str(getattr(reply_sender, "user_id", "") or "")
    _scheduler.notify(
        group_id,
        is_at=event.is_tome(),
        is_reply_to_bot=bool(reply_user_id and reply_user_id == bot.self_id),
        is_sticker=is_sticker,
        user_id=str(event.user_id),
        message_id=event.message_id,
    )


async def _auto_collect_stickers(bot: Bot, content: Content, is_sticker: bool, group_id: str) -> None:
    """Use vision metadata to collect incoming QQ stickers without model/tool permissions."""
    if not _sticker_auto_collect or _sticker_store is None or _vision_client is None:
        return
    if _sticker_auto_collect_only_stickers and not is_sticker:
        return
    now = time.monotonic()
    if now - _sticker_collect_last.get(group_id, 0.0) < _sticker_auto_collect_cooldown:
        return
    blocks = content if isinstance(content, list) else []
    image_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "image_ref"]
    if not image_blocks or len(_sticker_store.list_all()) >= _sticker_store.max_count:
        return
    _sticker_collect_last[group_id] = now
    for block in image_blocks:
        preview = block.get("path", "")
        original = block.get("original_path", preview)
        if not preview or not original:
            continue
        try:
            pair = await _vision_client.describe_sticker(preview, block.get("media_type", "image/jpeg"))
            if not pair:
                continue
            description, usage_hint = pair
            data = Path(original).read_bytes()
            sticker_id, is_new = _sticker_store.add(data, description, usage_hint, source="auto")
            if is_new:
                logger.info("auto sticker collected | group={} id={} description={}", group_id, sticker_id, description)
                path = _sticker_store.resolve_path(sticker_id)
                if path:
                    ctx = ToolContext(bot=bot, user_id="", group_id=group_id)
                    if await _deliver_sticker(ctx, path):
                        _sticker_store.record_send(sticker_id)
                        try:
                            await bot.send_group_msg(
                                group_id=int(group_id),
                                message=(
                                    f"已收录 {sticker_id}：{description}；"
                                    f"适合在{usage_hint}时使用。有不对的地方告诉我改。"
                                ),
                            )
                        except Exception:
                            logger.warning("auto sticker description send failed | id={}", sticker_id, exc_info=True)
            break
        except (OSError, ValueError):
            logger.debug("auto sticker collect skipped", exc_info=True)


# ── 群禁言监听 ──

ban_notice = on_notice(priority=1, block=False)


@ban_notice.handle()
async def handle_group_ban(bot: Bot, event: GroupBanNoticeEvent) -> None:
    if str(event.user_id) != bot.self_id:
        return
    group_id = str(event.group_id)
    if event.sub_type == "ban":
        _scheduler.mute(group_id)
        logger.warning("bot muted | group={} duration={}s", group_id, event.duration)
    elif event.sub_type == "lift_ban":
        _scheduler.unmute(group_id)
        logger.info("bot unmuted | group={}", group_id)


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

    # Security-sensitive commands bypass the LLM so dispatch is deterministic.
    command = user_content.strip() if isinstance(user_content, str) else ""
    if command in {"/登录网易云", "/网易云登录"}:
        if _music_client is None:
            await private_chat.finish(Message("网易云音乐模块未启用，请检查 [music] 配置。"))
        result = await MusicQrLoginTool(_music_client, _superusers).execute(ctx)
        await private_chat.finish(Message(result))
    if command in {"/检查网易云登录状态", "/检查登录状态"}:
        if _music_client is None:
            await private_chat.finish(Message("网易云音乐模块未启用，请检查 [music] 配置。"))
        result = await MusicLoginStatusTool(_music_client, _superusers).execute(ctx)
        await private_chat.finish(Message(result))

    sent_segments: set[str] = set()

    async def send_segment(text: str) -> None:
        if not text or text in sent_segments:
            return
        sent_segments.add(text)
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

    if reply and _sticker_sender is not None and not ctx.extra.get("sticker_sent"):
        await _sticker_sender.execute_random(ctx)
    if reply and reply not in sent_segments:
        await private_chat.finish(Message(reply))
