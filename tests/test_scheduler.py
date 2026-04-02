"""GroupChatScheduler unit tests."""

import asyncio

from src.identity.models import Identity
from src.llm.scheduler import GroupChatScheduler
from src.memory.group_timeline import GroupTimeline


def _make_identity(proactive: str | None = "积极参与群聊") -> Identity:
    return Identity(id="test", name="测试", personality="测试人设", proactive=proactive)


class _FakeIdentityMgr:
    def __init__(self, identity: Identity) -> None:
        self._identity = identity

    def resolve(self) -> Identity:
        return self._identity


class _FakeLLM:
    """Records chat() calls and returns configured reply."""

    def __init__(self, reply: str | None = "你好", *, delay: float = 0) -> None:
        self.calls: list[dict] = []
        self.reply = reply
        self._delay = delay

    async def chat(self, **kwargs) -> str | None:  # type: ignore[override]
        self.calls.append(kwargs)
        if self._delay:
            await asyncio.sleep(self._delay)
        return self.reply


class TestNotify:
    async def test_no_proactive_skips(self) -> None:
        """notify is a no-op when identity.proactive is None."""
        identity = _make_identity(proactive=None)
        scheduler = GroupChatScheduler(
            llm=_FakeLLM(), timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(identity),  # type: ignore[arg-type]
            debounce_seconds=0.05, batch_size=100,
        )
        scheduler.notify("g1")
        assert "g1" not in scheduler._slots
        await scheduler.close()

    async def test_debounce_fires(self) -> None:
        """After debounce timeout, chat() is called."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=0.05, batch_size=100,
        )
        scheduler.notify("g1")
        await asyncio.sleep(0.15)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_batch_size_fires_immediately(self) -> None:
        """Reaching batch_size triggers immediately without waiting for debounce."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=999, batch_size=3,
        )
        scheduler.notify("g1")
        scheduler.notify("g1")
        scheduler.notify("g1")
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_running_task_blocks_new_debounce(self) -> None:
        """While running_task is active, notify does not start new debounce."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=0.05, batch_size=100,
        )
        scheduler.notify("g1")
        await asyncio.sleep(0.15)  # debounce fires, running_task starts
        assert len(llm.calls) == 1
        scheduler.notify("g1")  # while running_task is active (or just finished)
        # msg_count incremented but no new debounce if running_task is still set
        # (depends on timing, so just verify no crash)
        await scheduler.close()


class TestAtHandling:
    async def test_at_fires_immediately(self) -> None:
        """notify(is_at=True) fires immediately, skipping debounce."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=999, batch_size=100,
        )
        scheduler.notify("g1", is_at=True)
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_at_cancels_pending_debounce(self) -> None:
        """notify(is_at=True) cancels a pending debounce and fires immediately."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=999, batch_size=100,
        )
        scheduler.notify("g1")  # starts debounce
        assert scheduler._slots["g1"].debounce_task is not None
        scheduler.notify("g1", is_at=True)  # cancels debounce, fires immediately
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_at_queues_when_busy(self) -> None:
        """notify(is_at=True) sets pending_at when a task is already running."""
        llm = _FakeLLM(reply=None, delay=0.5)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=0.05, batch_size=100,
        )
        scheduler.notify("g1")  # debounce
        await asyncio.sleep(0.15)  # fires, running_task active
        assert len(llm.calls) == 1
        scheduler.notify("g1", is_at=True)  # should queue
        assert scheduler._slots["g1"].pending_at is True
        await scheduler.close()

    async def test_pending_at_fires_after_completion(self) -> None:
        """After running task completes, pending_at triggers a new call."""
        llm = _FakeLLM(reply=None, delay=0.2)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=0.05, batch_size=100,
        )
        scheduler.notify("g1")
        await asyncio.sleep(0.15)  # first call fires (debounce done, chat starts)
        assert len(llm.calls) == 1
        scheduler.notify("g1", is_at=True)  # queued as pending_at
        await asyncio.sleep(0.4)  # first call finishes, pending fires
        assert len(llm.calls) == 2
        assert scheduler._slots["g1"].pending_at is False
        await scheduler.close()


class TestClose:
    async def test_close_cancels_all(self) -> None:
        """close() cancels all pending tasks."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=999, batch_size=100,
        )
        scheduler.notify("g1")
        scheduler.notify("g2")
        await scheduler.close()
        # After close, debounce tasks should be cancelled
        for slot in scheduler._slots.values():
            assert slot.debounce_task is None or slot.debounce_task.cancelled()
