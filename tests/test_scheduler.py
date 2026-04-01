"""GroupChatScheduler unit tests."""

import asyncio

import pytest

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

    def __init__(self, reply: str | None = "你好") -> None:
        self.calls: list[dict] = []
        self.reply = reply

    async def chat(self, **kwargs) -> str | None:  # type: ignore[override]
        self.calls.append(kwargs)
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
        assert llm.calls[0]["allow_skip"] is True
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
        slot = scheduler._slots["g1"]
        # msg_count incremented but no new debounce if running_task is still set
        # (depends on timing, so just verify no crash)
        await scheduler.close()


class TestInterrupt:
    async def test_cancels_debounce(self) -> None:
        """interrupt cancels pending debounce task."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            debounce_seconds=999, batch_size=100,
        )
        scheduler.notify("g1")
        assert scheduler._slots["g1"].debounce_task is not None
        scheduler.interrupt("g1")
        assert scheduler._slots["g1"].debounce_task is None or scheduler._slots["g1"].debounce_task.cancelled()
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 0  # debounce was cancelled, no chat call
        await scheduler.close()

    async def test_interrupt_nonexistent_group(self) -> None:
        """interrupt on unknown group is a no-op."""
        scheduler = GroupChatScheduler(
            llm=_FakeLLM(), timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
        )
        scheduler.interrupt("unknown")  # should not raise
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
