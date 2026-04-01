"""群聊统一调度器：debounce/batch 触发模型调用，@bot 抢占。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.memory.group_timeline import GroupTimeline
from src.tools.context import ToolContext

if TYPE_CHECKING:
    from nonebot.adapters.onebot.v11 import Bot

    from src.identity import IdentityManager
    from src.llm.client import LLMClient


class _GroupSlot:
    __slots__ = ("debounce_task", "interrupted", "msg_count", "running_task")

    def __init__(self) -> None:
        self.debounce_task: asyncio.Task[None] | None = None
        self.running_task: asyncio.Task[None] | None = None
        self.msg_count: int = 0
        self.interrupted: bool = False


class GroupChatScheduler:
    """群聊调度器：debounce 触发模型调用，@bot 抢占。"""

    def __init__(
        self,
        llm: LLMClient,
        timeline: GroupTimeline,
        identity_mgr: IdentityManager,
        debounce_seconds: float = 5.0,
        batch_size: int = 10,
    ) -> None:
        self._llm = llm
        self._timeline = timeline
        self._identity_mgr = identity_mgr
        self._debounce_seconds = debounce_seconds
        self._batch_size = batch_size
        self._slots: dict[str, _GroupSlot] = {}
        self._bot: Bot | None = None

    def set_bot(self, bot: Bot) -> None:
        self._bot = bot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify(self, group_id: str) -> None:
        """Called on every non-@ group message. Manages debounce/batch."""
        identity = self._identity_mgr.resolve()
        if identity.proactive is None:
            return

        slot = self._slots.setdefault(group_id, _GroupSlot())
        slot.msg_count += 1

        # If a chat() is running or @bot is being handled, don't schedule
        if slot.interrupted or (slot.running_task and not slot.running_task.done()):
            logger.debug("scheduler | group={} busy, skip (msgs={})", group_id, slot.msg_count)
            return

        # Cancel old debounce
        if slot.debounce_task and not slot.debounce_task.done():
            slot.debounce_task.cancel()

        # Batch full -> fire immediately
        if slot.msg_count >= self._batch_size:
            logger.info("scheduler | group={} batch full ({} msgs) -> fire", group_id, slot.msg_count)
            self._fire(group_id)
        else:
            logger.debug("scheduler | group={} debounce start (msgs={})", group_id, slot.msg_count)
            slot.debounce_task = asyncio.create_task(self._debounce(group_id))

    def interrupt(self, group_id: str) -> None:
        """Called when @bot. Cancels debounce and running task for this group."""
        slot = self._slots.get(group_id)
        if not slot:
            return

        if slot.debounce_task and not slot.debounce_task.done():
            slot.debounce_task.cancel()
            slot.debounce_task = None
            logger.debug("scheduler | group={} debounce cancelled by @bot", group_id)

        if slot.running_task and not slot.running_task.done():
            slot.running_task.cancel()
            slot.running_task = None
            logger.info("scheduler | group={} running task cancelled by @bot", group_id)

        slot.msg_count = 0
        slot.interrupted = True

    def release(self, group_id: str) -> None:
        """Called after @bot chat completes. Re-enables scheduler for this group."""
        slot = self._slots.get(group_id)
        if slot:
            slot.interrupted = False

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

    async def _debounce(self, group_id: str) -> None:
        try:
            await asyncio.sleep(self._debounce_seconds)
            slot = self._slots.get(group_id)
            if slot and slot.msg_count > 0:
                logger.info("scheduler | group={} debounce fired ({} msgs)", group_id, slot.msg_count)
                self._fire(group_id)
        except asyncio.CancelledError:
            pass

    def _fire(self, group_id: str) -> None:
        slot = self._slots.get(group_id)
        if not slot:
            return
        slot.msg_count = 0
        slot.running_task = asyncio.create_task(self._do_chat(group_id))
        slot.running_task.add_done_callback(lambda _: None)

    async def _send_to_group(self, group_id: str, text: str) -> None:
        """Send a text message to a group. No-op if bot is not set."""
        if self._bot:
            from nonebot.adapters.onebot.v11 import Message

            await self._bot.send_group_msg(group_id=int(group_id), message=Message(text))

    async def _do_chat(self, group_id: str) -> None:
        slot = self._slots.get(group_id)
        try:
            identity = self._identity_mgr.resolve()
            session_id = f"group_{group_id}"
            ctx = ToolContext(bot=self._bot, user_id="", group_id=group_id)

            async def on_segment(text: str) -> None:
                await self._send_to_group(group_id, text)

            reply = await self._llm.chat(
                session_id=session_id,
                user_id="",
                user_text="",
                identity=identity,
                group_id=group_id,
                ctx=ctx,
                on_segment=on_segment if self._bot else None,
                allow_skip=True,
            )

            if reply:
                await self._send_to_group(group_id, reply)

        except asyncio.CancelledError:
            logger.debug("scheduler | group={} chat cancelled", group_id)
        except Exception:
            logger.exception("scheduler | group={} chat error", group_id)
        finally:
            if slot:
                slot.running_task = None
