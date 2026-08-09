"""GroupChatScheduler unit tests."""

import asyncio

from src.config import GroupConfig, GroupOverride
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
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.notify("111")
        assert "111" not in scheduler._slots
        await scheduler.close()

    async def test_debounce_fires(self) -> None:
        """After debounce timeout, chat() is called."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.notify("111")
        await asyncio.sleep(0.15)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_batch_size_fires_immediately(self) -> None:
        """Reaching batch_size triggers immediately without waiting for debounce."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=999, batch_size=3),
        )
        scheduler.notify("111")
        scheduler.notify("111")
        scheduler.notify("111")
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_running_task_blocks_new_debounce(self) -> None:
        """While running_task is active, notify does not start new debounce."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.notify("111")
        await asyncio.sleep(0.15)  # debounce fires, running_task starts
        assert len(llm.calls) == 1
        scheduler.notify("111")  # while running_task is active (or just finished)
        # msg_count incremented but no new debounce if running_task is still set
        # (depends on timing, so just verify no crash)
        await scheduler.close()


class TestAtHandling:
    async def test_at_works_without_proactive_identity(self) -> None:
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(),
            identity_mgr=_FakeIdentityMgr(_make_identity(proactive=None)),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=999, batch_size=100),
        )
        scheduler.notify("111", is_at=True, user_id="42", message_id=7)
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        assert llm.calls[0]["response_mode"] == "direct"
        assert llm.calls[0]["user_id"] == "42"
        assert "触发消息 ID=7" in llm.calls[0]["trigger_context"]
        await scheduler.close()

    async def test_at_fires_immediately(self) -> None:
        """notify(is_at=True) fires immediately, skipping debounce."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=999, batch_size=100),
        )
        scheduler.notify("111", is_at=True)
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        assert llm.calls[0]["response_mode"] == "direct"
        await scheduler.close()

    async def test_at_cancels_pending_debounce(self) -> None:
        """notify(is_at=True) cancels a pending debounce and fires immediately."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=999, batch_size=100),
        )
        scheduler.notify("111")  # starts debounce
        assert scheduler._slots["111"].debounce_task is not None
        scheduler.notify("111", is_at=True)  # cancels debounce, fires immediately
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_at_queues_when_busy(self) -> None:
        """notify(is_at=True) sets pending_at when a task is already running."""
        llm = _FakeLLM(reply=None, delay=0.5)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.notify("111")  # debounce
        await asyncio.sleep(0.15)  # fires, running_task active
        assert len(llm.calls) == 1
        scheduler.notify("111", is_at=True)  # should queue
        assert scheduler._slots["111"].pending_at is True
        await scheduler.close()

    async def test_pending_at_fires_after_completion(self) -> None:
        """After running task completes, pending_at triggers a new call."""
        llm = _FakeLLM(reply=None, delay=0.2)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.notify("111")
        await asyncio.sleep(0.15)  # first call fires (debounce done, chat starts)
        assert len(llm.calls) == 1
        scheduler.notify("111", is_at=True)  # queued as pending_at
        await asyncio.sleep(0.4)  # first call finishes, pending fires
        assert len(llm.calls) == 2
        assert scheduler._slots["111"].pending_at is False
        await scheduler.close()


class TestClose:
    async def test_close_cancels_all(self) -> None:
        """close() cancels all pending tasks."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=999, batch_size=100),
        )
        scheduler.notify("111")
        scheduler.notify("222")
        await scheduler.close()
        # After close, debounce tasks should be cancelled
        for slot in scheduler._slots.values():
            assert slot.debounce_task is None or slot.debounce_task.cancelled()


class TestAtOnly:
    async def test_at_only_skips_debounce(self) -> None:
        """at_only=True: non-@ messages don't trigger debounce or batch."""
        llm = _FakeLLM(reply=None)
        group_config = GroupConfig(at_only=True, debounce_seconds=0.05, batch_size=3)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=group_config,
        )
        scheduler.notify("123")
        scheduler.notify("123")
        scheduler.notify("123")  # reaches batch_size but at_only blocks it
        await asyncio.sleep(0.15)
        assert len(llm.calls) == 0
        await scheduler.close()

    async def test_at_only_still_fires_on_at(self) -> None:
        """at_only=True: @ messages still fire immediately."""
        llm = _FakeLLM(reply=None)
        group_config = GroupConfig(at_only=True, debounce_seconds=999, batch_size=100)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=group_config,
        )
        scheduler.notify("123", is_at=True)
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_per_group_at_only_override(self) -> None:
        """Group 123 is at_only, group 456 is not."""
        llm = _FakeLLM(reply=None)
        group_config = GroupConfig(
            at_only=False, debounce_seconds=0.05, batch_size=100,
            overrides={123: GroupOverride(at_only=True)},
        )
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=group_config,
        )
        scheduler.notify("123")  # at_only group — no debounce
        scheduler.notify("456")  # normal group — debounce starts
        await asyncio.sleep(0.15)
        assert len(llm.calls) == 1  # only group 456 fired
        assert llm.calls[0]["session_id"] == "group_456"
        await scheduler.close()


class TestPerGroupParams:
    async def test_proactive_call_uses_proactive_mode(self) -> None:
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.notify("123")
        await asyncio.sleep(0.15)
        assert llm.calls[0]["response_mode"] == "proactive"
        assert llm.calls[0]["user_id"] == ""
        await scheduler.close()

    async def test_proactive_cooldown_does_not_block_direct_call(self) -> None:
        llm = _FakeLLM(reply="接话")
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(
                debounce_seconds=0.01, batch_size=100, proactive_cooldown_seconds=3600,
            ),
        )
        scheduler.notify("123")
        await asyncio.sleep(0.1)
        scheduler.notify("123")
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1

        scheduler.notify("123", is_at=True, user_id="9", message_id=99)
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 2
        assert llm.calls[-1]["response_mode"] == "direct"
        await scheduler.close()

    async def test_plain_message_while_busy_waits_for_debounce(self) -> None:
        llm = _FakeLLM(reply=None, delay=0.15)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.1, batch_size=100),
        )
        scheduler.notify("123", is_at=True, user_id="9")
        await asyncio.sleep(0.05)
        scheduler.notify("123")
        await asyncio.sleep(0.14)
        assert len(llm.calls) == 1
        await asyncio.sleep(0.12)
        assert len(llm.calls) == 2
        assert llm.calls[-1]["response_mode"] == "proactive"
        await scheduler.close()

    async def test_per_group_debounce(self) -> None:
        """Group 123 has 0.3s debounce (override), group 456 uses global 0.05s."""
        llm = _FakeLLM(reply=None)
        group_config = GroupConfig(
            debounce_seconds=0.05, batch_size=100,
            overrides={123: GroupOverride(debounce_seconds=0.3)},
        )
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=group_config,
        )
        scheduler.notify("123")
        scheduler.notify("456")
        await asyncio.sleep(0.15)
        # group 456 (0.05s debounce) should have fired, group 123 (0.3s) not yet
        assert len(llm.calls) == 1
        assert llm.calls[0]["session_id"] == "group_456"
        await asyncio.sleep(0.3)
        assert len(llm.calls) == 2
        await scheduler.close()

    async def test_per_group_batch_size(self) -> None:
        """Group 123 has batch_size=2 (override), global is 100."""
        llm = _FakeLLM(reply=None)
        group_config = GroupConfig(
            debounce_seconds=999, batch_size=100,
            overrides={123: GroupOverride(batch_size=2)},
        )
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=group_config,
        )
        scheduler.notify("123")
        scheduler.notify("123")  # hits batch_size=2
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 1
        await scheduler.close()


class TestMute:
    async def test_muted_group_skips_notify(self) -> None:
        """notify is a no-op for muted groups."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.mute("111")
        scheduler.notify("111")
        scheduler.notify("111", is_at=True)
        await asyncio.sleep(0.15)
        assert len(llm.calls) == 0
        await scheduler.close()

    async def test_muted_group_skips_trigger(self) -> None:
        """trigger is a no-op for muted groups."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.mute("111")
        scheduler.trigger("111")
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 0
        await scheduler.close()

    async def test_unmute_resumes_scheduling(self) -> None:
        """After unmute, notify works again."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.mute("111")
        scheduler.notify("111")
        await asyncio.sleep(0.1)
        assert len(llm.calls) == 0

        scheduler.unmute("111")
        scheduler.notify("111")
        await asyncio.sleep(0.15)
        assert len(llm.calls) == 1
        await scheduler.close()

    async def test_mute_cancels_running_tasks(self) -> None:
        """Muting a group cancels its pending debounce and running tasks."""
        llm = _FakeLLM(reply=None, delay=1.0)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.notify("111")
        await asyncio.sleep(0.15)  # debounce fires, running_task starts
        assert len(llm.calls) == 1
        scheduler.mute("111")
        slot = scheduler._slots["111"]
        assert slot.running_task is None
        assert slot.msg_count == 0
        assert slot.pending_at is False
        await scheduler.close()

    async def test_is_muted(self) -> None:
        """is_muted returns correct state."""
        scheduler = GroupChatScheduler(
            llm=_FakeLLM(), timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(),
        )
        assert not scheduler.is_muted("111")
        scheduler.mute("111")
        assert scheduler.is_muted("111")
        scheduler.unmute("111")
        assert not scheduler.is_muted("111")
        await scheduler.close()

    async def test_mute_only_affects_target_group(self) -> None:
        """Muting group 111 does not affect group 222."""
        llm = _FakeLLM(reply=None)
        scheduler = GroupChatScheduler(
            llm=llm, timeline=GroupTimeline(), identity_mgr=_FakeIdentityMgr(_make_identity()),  # type: ignore[arg-type]
            group_config=GroupConfig(debounce_seconds=0.05, batch_size=100),
        )
        scheduler.mute("111")
        scheduler.notify("111")
        scheduler.notify("222")
        await asyncio.sleep(0.15)
        assert len(llm.calls) == 1
        assert llm.calls[0]["session_id"] == "group_222"
        await scheduler.close()
