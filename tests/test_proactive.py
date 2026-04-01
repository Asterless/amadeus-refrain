"""ProactiveEvaluator 单元测试。"""

import time

import pytest

from src.identity.models import Identity
from src.llm.proactive import ProactiveDecision, ProactiveEvaluator, _GroupBatch
from src.memory.group_timeline import GroupTimeline


def _make_identity(proactive: str | None = None) -> Identity:
    return Identity(
        id="test",
        name="测试",
        personality="测试人设",
        proactive=proactive,
    )


@pytest.fixture
def timeline() -> GroupTimeline:
    tl = GroupTimeline(max_messages=50)
    for i in range(5):
        tl.add("g1", role="user", speaker=f"用户{i}(100{i})", content=f"消息{i}")
    return tl


class TestNotify:
    def test_disabled_skips(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", enabled=False)
        identity = _make_identity(proactive="规则")
        ev.notify("g1", identity, on_reply=lambda d: None)  # type: ignore[arg-type]
        assert "g1" not in ev._batches

    def test_no_proactive_skips(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u")
        identity = _make_identity(proactive=None)
        ev.notify("g1", identity, on_reply=lambda d: None)  # type: ignore[arg-type]
        assert "g1" not in ev._batches

    def test_cooldown_skips(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", cooldown=60)
        identity = _make_identity(proactive="规则")
        ev._last_proactive["g1"] = time.monotonic()
        ev.notify("g1", identity, on_reply=lambda d: None)  # type: ignore[arg-type]
        assert "g1" not in ev._batches

    @pytest.mark.asyncio
    async def test_increments_msg_count(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u")
        identity = _make_identity(proactive="规则")

        async def noop(d: ProactiveDecision) -> None:
            pass

        ev.notify("g1", identity, on_reply=noop)
        assert ev._batches["g1"].msg_count == 1
        ev.notify("g1", identity, on_reply=noop)
        assert ev._batches["g1"].msg_count == 2
        await ev.close()

    def test_sets_pending_when_evaluating(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u")
        identity = _make_identity(proactive="规则")

        async def noop(d: ProactiveDecision) -> None:
            pass

        # 模拟正在评估
        batch = _GroupBatch()
        batch.evaluating = True
        ev._batches["g1"] = batch

        ev.notify("g1", identity, on_reply=noop)
        assert batch.pending is True


class TestBuildDecisionPrompt:
    def test_prompt_content(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", context_lines=3)
        identity = _make_identity(proactive="只在有人求助时插话")
        system, messages = ev.build_decision_prompt("g1", identity)
        assert "测试人设" in system[0]["text"]
        assert "只在有人求助时插话" in system[0]["text"]
        user_text = messages[0]["content"]
        assert "消息2" in user_text
        assert "消息3" in user_text
        assert "消息4" in user_text

    def test_prompt_respects_context_lines(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", context_lines=2)
        identity = _make_identity(proactive="规则")
        _, messages = ev.build_decision_prompt("g1", identity)
        user_text = messages[0]["content"]
        assert "消息3" in user_text
        assert "消息4" in user_text
        assert "消息0" not in user_text


class TestParseDecision:
    def test_valid_reply_true(self) -> None:
        text = '{"reply": true, "reason": "有人问物理", "reply_to": "张三"}'
        result = ProactiveEvaluator._parse_decision(text)
        assert result == ProactiveDecision(reason="有人问物理", reply_to="张三")

    def test_valid_reply_false(self) -> None:
        text = '{"reply": false, "reason": "", "reply_to": ""}'
        assert ProactiveEvaluator._parse_decision(text) is None

    def test_json_embedded_in_text(self) -> None:
        text = '好的，我来判断一下\n{"reply": true, "reason": "求助", "reply_to": "大家"}\n'
        result = ProactiveEvaluator._parse_decision(text)
        assert result is not None
        assert result["reason"] == "求助"

    def test_invalid_json(self) -> None:
        assert ProactiveEvaluator._parse_decision("YES") is None

    def test_missing_fields(self) -> None:
        text = '{"reply": true}'
        result = ProactiveEvaluator._parse_decision(text)
        assert result is not None
        assert result["reason"] == ""
        assert result["reply_to"] == ""


class TestBatchTrigger:
    @pytest.mark.asyncio
    async def test_debounce_fires_after_timeout(self, timeline: GroupTimeline) -> None:
        """消息间隙后触发评估（通过 debounce task 存在来验证）。"""
        ev = ProactiveEvaluator(
            timeline=timeline, model="m", api_key="k", base_url="u",
            batch_timeout=0.1, batch_size=100,
        )
        identity = _make_identity(proactive="规则")

        async def noop(d: ProactiveDecision) -> None:
            pass

        ev.notify("g1", identity, on_reply=noop)
        batch = ev._batches["g1"]
        assert batch.debounce_task is not None
        assert not batch.debounce_task.done()
        await ev.close()

    @pytest.mark.asyncio
    async def test_batch_size_fires_immediately(self, timeline: GroupTimeline) -> None:
        """达到 batch_size 时立即触发（evaluating 变为 True）。"""
        ev = ProactiveEvaluator(
            timeline=timeline, model="m", api_key="k", base_url="u",
            batch_timeout=999, batch_size=3,
        )
        identity = _make_identity(proactive="规则")

        async def noop(d: ProactiveDecision) -> None:
            pass

        ev.notify("g1", identity, on_reply=noop)
        ev.notify("g1", identity, on_reply=noop)
        assert not ev._batches["g1"].evaluating  # 还没到 batch_size
        ev.notify("g1", identity, on_reply=noop)  # 第 3 条，触发
        # evaluating 会在 task 启动后设为 True
        # msg_count 被重置为 0
        assert ev._batches["g1"].msg_count == 0
        await ev.close()
