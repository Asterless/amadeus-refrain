"""ProactiveEvaluator 单元测试。"""

import asyncio
import time

import pytest

from src.identity.models import Identity
from src.llm.proactive import ProactiveDecision, ProactiveEvaluator
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


class TestShouldEvaluate:
    def test_no_proactive_rule(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u")
        identity = _make_identity(proactive=None)
        assert ev.should_evaluate("g1", identity) is False

    def test_has_proactive_rule(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u")
        identity = _make_identity(proactive="随便插话")
        assert ev.should_evaluate("g1", identity) is True

    def test_cooldown_blocks(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", cooldown=60)
        identity = _make_identity(proactive="随便插话")
        ev._last_proactive["g1"] = time.monotonic()  # 刚刚插话过
        assert ev.should_evaluate("g1", identity) is False

    def test_cooldown_expired(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", cooldown=60)
        identity = _make_identity(proactive="随便插话")
        ev._last_proactive["g1"] = time.monotonic() - 120  # 很久以前
        assert ev.should_evaluate("g1", identity) is True

    @pytest.mark.asyncio
    async def test_locked_group_blocks(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u")
        identity = _make_identity(proactive="随便插话")
        lock = asyncio.Lock()
        await lock.acquire()
        ev._locks["g1"] = lock
        assert ev.should_evaluate("g1", identity) is False
        lock.release()


class TestBuildDecisionPrompt:
    def test_prompt_content(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", context_lines=3)
        identity = _make_identity(proactive="只在有人求助时插话")
        system, messages = ev.build_decision_prompt("g1", identity)
        assert "测试人设" in system[0]["text"]
        assert "只在有人求助时插话" in system[0]["text"]
        # messages 的 user content 应包含最近 3 条消息
        user_text = messages[0]["content"]
        assert "消息2" in user_text
        assert "消息3" in user_text
        assert "消息4" in user_text

    def test_prompt_respects_context_lines(self, timeline: GroupTimeline) -> None:
        ev = ProactiveEvaluator(timeline=timeline, model="m", api_key="k", base_url="u", context_lines=2)
        identity = _make_identity(proactive="规则")
        _, messages = ev.build_decision_prompt("g1", identity)
        user_text = messages[0]["content"]
        # 只取最近 2 条
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
