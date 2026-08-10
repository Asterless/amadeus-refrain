"""群聊统一调度器：debounce/batch/@ 触发模型调用，统一队列。"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from loguru import logger

from src.config import GroupConfig
from src.llm.client import _RATE_LIMIT_BASE_DELAY, _RATE_LIMIT_MAX_RETRIES, RateLimitError
from src.memory.group_timeline import GroupTimeline
from src.tools.context import ToolContext
from src.tools.sticker_tools import SendStickerTool

if TYPE_CHECKING:
    from nonebot.adapters.onebot.v11 import Bot

    from src.identity import IdentityManager
    from src.llm.client import LLMClient


class _GroupSlot:
    __slots__ = (
        "actor_id",
        "debounce_task",
        "fire_at",
        "last_proactive_reply_at",
        "msg_count",
        "pending_at",
        "proactive_reply_times",
        "running_task",
        "trigger_message_id",
        "trigger_reason",
    )

    def __init__(self) -> None:
        self.debounce_task: asyncio.Task[None] | None = None
        self.running_task: asyncio.Task[None] | None = None
        self.msg_count: int = 0
        self.pending_at: bool = False
        self.fire_at: bool = False
        self.actor_id: str = ""
        self.trigger_message_id: int | None = None
        self.trigger_reason: str = ""
        self.last_proactive_reply_at: float = 0.0
        self.proactive_reply_times: list[float] = []


class GroupChatScheduler:
    """群聊统一调度器：debounce/batch/@触发模型调用。"""

    def __init__(
        self,
        llm: LLMClient,
        timeline: GroupTimeline,
        identity_mgr: IdentityManager,
        group_config: GroupConfig,
        always_describe_images: bool = False,
        reply_on_sticker: bool = False,
        auto_sticker_sender: SendStickerTool | None = None,
    ) -> None:
        self._llm = llm
        self._timeline = timeline
        self._identity_mgr = identity_mgr
        self._group_config = group_config
        self._always_describe_images = always_describe_images
        self._reply_on_sticker = reply_on_sticker
        self._auto_sticker_sender = auto_sticker_sender
        self._slots: dict[str, _GroupSlot] = {}
        self._bot: Bot | None = None
        self._muted_groups: set[str] = set()

    def set_bot(self, bot: Bot) -> None:
        self._bot = bot

    # ------------------------------------------------------------------
    # Mute management
    # ------------------------------------------------------------------

    def mute(self, group_id: str) -> None:
        """Mark group as muted — cancel pending tasks, block future fires."""
        self._muted_groups.add(group_id)
        slot = self._slots.get(group_id)
        if slot:
            for task in (slot.debounce_task, slot.running_task):
                if task and not task.done():
                    task.cancel()
            slot.debounce_task = None
            slot.running_task = None
            slot.msg_count = 0
            slot.pending_at = False
            slot.fire_at = False
            slot.actor_id = ""
            slot.trigger_message_id = None
            slot.trigger_reason = ""
        logger.info("scheduler | group={} muted, tasks cancelled", group_id)

    def unmute(self, group_id: str) -> None:
        """Unmark group as muted — resume normal scheduling."""
        self._muted_groups.discard(group_id)
        logger.info("scheduler | group={} unmuted", group_id)

    def is_muted(self, group_id: str) -> bool:
        return group_id in self._muted_groups

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify(
        self,
        group_id: str,
        *,
        is_at: bool = False,
        is_reply_to_bot: bool = False,
        is_sticker: bool = False,
        user_id: str = "",
        message_id: int | None = None,
    ) -> None:
        """Called on every group message. Manages debounce/batch."""
        if group_id in self._muted_groups:
            return
        resolved = self._group_config.resolve(int(group_id))

        slot = self._slots.setdefault(group_id, _GroupSlot())
        slot.msg_count += 1

        direct_reason = ""
        if is_at:
            direct_reason = "用户@了你"
        elif is_reply_to_bot:
            direct_reason = "用户回复了你的消息"
        elif is_sticker and self._reply_on_sticker:
            direct_reason = "用户发送了图片表情，配置要求接话"

        if direct_reason:
            if slot.running_task and not slot.running_task.done():
                slot.pending_at = True
                self._set_direct_trigger(slot, direct_reason, user_id, message_id)
                logger.debug("scheduler | group={} direct queued (task running)", group_id)
                return
            if slot.debounce_task and not slot.debounce_task.done():
                slot.debounce_task.cancel()
            self._set_direct_trigger(slot, direct_reason, user_id, message_id)
            logger.info("scheduler | group={} direct={} -> fire", group_id, direct_reason)
            self._fire(group_id)
            return

        identity = self._identity_mgr.resolve()
        if identity.proactive is None:
            return

        # at_only mode: only respond to @ messages
        if resolved.at_only:
            logger.debug("scheduler | group={} at_only, skip (msgs={})", group_id, slot.msg_count)
            return

        if slot.running_task and not slot.running_task.done():
            logger.debug("scheduler | group={} busy, skip (msgs={})", group_id, slot.msg_count)
            return

        if slot.debounce_task and not slot.debounce_task.done():
            slot.debounce_task.cancel()

        if slot.msg_count >= resolved.batch_size:
            if self._proactive_allowed(group_id, slot):
                logger.info("scheduler | group={} batch full ({} msgs) -> fire", group_id, slot.msg_count)
                self._set_proactive_trigger(slot, "消息数量达到 batch_size")
                self._fire(group_id)
            else:
                slot.msg_count = 0
        else:
            logger.debug("scheduler | group={} debounce start (msgs={})", group_id, slot.msg_count)
            slot.fire_at = False
            slot.debounce_task = asyncio.create_task(
                self._debounce(group_id, resolved.debounce_seconds)
            )

    def trigger(self, group_id: str) -> None:
        """Immediately fire a chat for this group (no debounce). Used at startup."""
        if group_id in self._muted_groups:
            return
        identity = self._identity_mgr.resolve()
        if identity.proactive is None:
            return
        slot = self._slots.setdefault(group_id, _GroupSlot())
        if slot.running_task and not slot.running_task.done():
            return
        logger.info("scheduler | group={} trigger (startup)", group_id)
        self._set_proactive_trigger(slot, "启动时检查历史消息")
        self._fire(group_id)

    async def close(self) -> None:
        """Cancel all pending tasks on shutdown."""
        tasks: list[asyncio.Task[None]] = []
        for slot in self._slots.values():
            for task in (slot.debounce_task, slot.running_task):
                if task and not task.done():
                    task.cancel()
                    tasks.append(task)
        # Let cancelled tasks finish their CancelledError handling
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _debounce(self, group_id: str, seconds: float) -> None:
        try:
            await asyncio.sleep(seconds)
            slot = self._slots.get(group_id)
            if slot and slot.msg_count > 0:
                if self._proactive_allowed(group_id, slot):
                    logger.info("scheduler | group={} debounce fired ({} msgs)", group_id, slot.msg_count)
                    self._set_proactive_trigger(slot, "群聊安静后主动判断是否接话")
                    self._fire(group_id)
                else:
                    slot.msg_count = 0
        except asyncio.CancelledError:
            pass

    def _fire(self, group_id: str) -> None:
        slot = self._slots.get(group_id)
        if not slot:
            return
        slot.msg_count = 0
        slot.running_task = asyncio.create_task(self._do_chat(group_id))
        slot.running_task.add_done_callback(lambda _: None)

    @staticmethod
    def _set_direct_trigger(
        slot: _GroupSlot,
        reason: str,
        user_id: str,
        message_id: int | None,
    ) -> None:
        slot.fire_at = True
        slot.actor_id = user_id
        slot.trigger_message_id = message_id
        slot.trigger_reason = reason

    @staticmethod
    def _set_proactive_trigger(slot: _GroupSlot, reason: str) -> None:
        slot.fire_at = False
        slot.actor_id = ""
        slot.trigger_message_id = None
        slot.trigger_reason = reason

    def _proactive_allowed(self, group_id: str, slot: _GroupSlot) -> bool:
        resolved = self._group_config.resolve(int(group_id))
        now = time.monotonic()
        if (
            resolved.proactive_cooldown_seconds > 0
            and now - slot.last_proactive_reply_at < resolved.proactive_cooldown_seconds
        ):
            logger.debug("scheduler | group={} proactive cooldown active", group_id)
            return False

        cutoff = now - 3600
        slot.proactive_reply_times = [ts for ts in slot.proactive_reply_times if ts >= cutoff]
        limit = resolved.proactive_max_replies_per_hour
        if limit > 0 and len(slot.proactive_reply_times) >= limit:
            logger.debug("scheduler | group={} proactive hourly limit reached", group_id)
            return False
        return True

    async def _send_to_group(self, group_id: str, text: str) -> None:
        """Send a text message to a group with retry on failure."""
        if not self._bot:
            return
        from nonebot.adapters.onebot.v11 import Message
        from nonebot.adapters.onebot.v11.exception import ActionFailed

        delay = 2.0
        max_delay = 60.0
        while True:
            if group_id in self._muted_groups:
                logger.warning("scheduler | group={} muted, dropping message", group_id)
                return
            try:
                await self._bot.send_group_msg(group_id=int(group_id), message=Message(text))
                return
            except ActionFailed as e:
                logger.warning(
                    "scheduler | group={} send failed: {} | retry in {}s",
                    group_id, e.info.get("wording") or e.info.get("message", str(e)), delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    async def _do_chat(self, group_id: str) -> None:
        slot = self._slots.get(group_id)
        is_direct = bool(slot and slot.fire_at)
        actor_id = slot.actor_id if slot else ""
        trigger_message_id = slot.trigger_message_id if slot else None
        trigger_reason = slot.trigger_reason if slot else ""
        describe_images = self._always_describe_images or is_direct
        sent_segments: set[str] = set()
        try:
            for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    identity = self._identity_mgr.resolve()
                    session_id = f"group_{group_id}"
                    ctx = ToolContext(bot=self._bot, user_id=actor_id, group_id=group_id)
                    trigger_context = trigger_reason
                    if actor_id:
                        trigger_context += f"；触发者 QQ={actor_id}"
                    if trigger_message_id is not None:
                        trigger_context += f"；触发消息 ID={trigger_message_id}"

                    async def on_segment(text: str) -> None:
                        if not text or text in sent_segments:
                            return
                        sent_segments.add(text)
                        await self._send_to_group(group_id, text)

                    reply = await self._llm.chat(
                        session_id=session_id,
                        user_id=actor_id,
                        user_content="",
                        identity=identity,
                        group_id=group_id,
                        ctx=ctx,
                        on_segment=on_segment if self._bot else None,
                        describe_images=describe_images,
                        response_mode="direct" if is_direct else "proactive",
                        trigger_context=trigger_context,
                    )

                    if reply and reply not in sent_segments:
                        await self._send_to_group(group_id, reply)
                        sent_segments.add(reply)
                        if not is_direct and slot:
                            now = time.monotonic()
                            slot.last_proactive_reply_at = now
                            slot.proactive_reply_times.append(now)
                    if reply and self._auto_sticker_sender and not ctx.extra.get("sticker_sent"):
                        await self._auto_sticker_sender.execute_random(ctx)
                    return

                except RateLimitError:
                    if attempt >= _RATE_LIMIT_MAX_RETRIES:
                        logger.error(
                            "scheduler | group={} rate limit exhausted after {} retries",
                            group_id, _RATE_LIMIT_MAX_RETRIES,
                        )
                        return
                    delay = _RATE_LIMIT_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "scheduler | group={} rate limited, retry {}/{} in {:.0f}s (will include new messages)",
                        group_id, attempt + 1, _RATE_LIMIT_MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)

        except asyncio.CancelledError:
            logger.debug("scheduler | group={} chat cancelled", group_id)
        except Exception:
            logger.exception("scheduler | group={} chat error", group_id)
        finally:
            if slot:
                slot.running_task = None
                if slot.pending_at:
                    slot.pending_at = False
                    self._fire(group_id)
                elif slot.msg_count > 0:
                    resolved = self._group_config.resolve(int(group_id))
                    if slot.debounce_task and not slot.debounce_task.done():
                        slot.debounce_task.cancel()
                    self._set_proactive_trigger(slot, "回复期间收到新消息，等待群聊安静")
                    slot.debounce_task = asyncio.create_task(
                        self._debounce(group_id, resolved.debounce_seconds)
                    )
