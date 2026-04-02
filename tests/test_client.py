"""Tests for LLMClient compact logic, circuit breaker, and micro compact."""

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.identity.models import Identity
from src.llm.client import LLMClient
from src.llm.prompt import PromptBuilder
from src.memory.group_timeline import GroupTimeline
from src.memory.memo_store import MemoStore
from src.memory.short_term import ShortTermMemory
from src.tools.registry import ToolRegistry

_IDENTITY = Identity(id="t", name="Bot", personality="p")


@pytest.fixture
def prompt() -> PromptBuilder:
    pb = PromptBuilder(instruction="test")
    pb.build_static(_IDENTITY, bot_self_id="999")
    return pb


@pytest.fixture
def short_term() -> ShortTermMemory:
    return ShortTermMemory()


@pytest.fixture
def tools() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def timeline() -> GroupTimeline:
    return GroupTimeline(max_messages=200)


@pytest.fixture
async def memo_store(tmp_path) -> MemoStore:
    s = MemoStore(base_dir=str(tmp_path))
    await s.startup()
    return s


async def _client(
    prompt: PromptBuilder,
    short_term: ShortTermMemory,
    tools: ToolRegistry,
    timeline: GroupTimeline | None = None,
    memo_store: MemoStore | None = None,
    max_compact_failures: int = 3,
) -> AsyncIterator[LLMClient]:
    c = LLMClient(
        base_url="http://fake",
        api_key="sk-fake",
        model="test-model",
        prompt_builder=prompt,
        short_term=short_term,
        tools=tools,
        max_context_tokens=100_000,
        micro_ratio=0.6,
        full_ratio=0.8,
        max_compact_failures=max_compact_failures,
        group_timeline=timeline,
        memo_store=memo_store,
    )
    try:
        yield c
    finally:
        await c.close()


def _fill_messages(short_term: ShortTermMemory, sid: str, count: int = 8) -> None:
    for i in range(count):
        short_term.add(sid, "user" if i % 2 == 0 else "assistant", f"msg {i}")


MOCK_RESULT = {"text": "summary", "tool_uses": [], "input_tokens": 100}


# ---------------------------------------------------------------------------
# Micro compact
# ---------------------------------------------------------------------------


async def test_micro_compact_private(prompt, short_term, tools) -> None:
    async for client in _client(prompt, short_term, tools):
        sid = "private_100"
        _fill_messages(short_term, sid)
        assert len(short_term.get(sid)) == 8
        client._micro_compact_private(sid)
        assert len(short_term.get(sid)) == 6  # dropped 2 (8 // 4)


async def test_micro_compact_private_too_few(prompt, short_term, tools) -> None:
    async for client in _client(prompt, short_term, tools):
        sid = "private_100"
        short_term.add(sid, "user", "hello")
        client._micro_compact_private(sid)
        assert len(short_term.get(sid)) == 1


async def test_micro_compact_group(prompt, short_term, tools, timeline) -> None:
    async for client in _client(prompt, short_term, tools, timeline=timeline):
        gid = "12345"
        for i in range(12):
            timeline.add(gid, role="user", content=f"msg {i}", speaker=f"user({i})")
        assert len(timeline.get_messages(gid)) == 12
        client._micro_compact_group(gid)
        assert len(timeline.get_messages(gid)) == 9  # dropped 3 (12 // 4)


# ---------------------------------------------------------------------------
# Circuit breaker — private
# ---------------------------------------------------------------------------


async def test_private_circuit_breaker_activates(prompt, short_term, tools) -> None:
    async for client in _client(prompt, short_term, tools, max_compact_failures=2):
        client._private_compact_failures = 2
        _fill_messages(short_term, "private_100")

        with patch("src.llm.client._call_api", new_callable=AsyncMock) as mock_api:
            await client._compact("private_100")
            mock_api.assert_not_called()


async def test_private_compact_resets_on_success(prompt, short_term, tools) -> None:
    async for client in _client(prompt, short_term, tools):
        client._private_compact_failures = 1
        _fill_messages(short_term, "private_100")

        with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=MOCK_RESULT):
            await client._compact("private_100")
        assert client._private_compact_failures == 0


async def test_private_compact_increments_on_failure(prompt, short_term, tools) -> None:
    async for client in _client(prompt, short_term, tools):
        _fill_messages(short_term, "private_100")

        with patch("src.llm.client._call_api", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            await client._compact("private_100")
        assert client._private_compact_failures == 1


# ---------------------------------------------------------------------------
# Circuit breaker — group (independent from private)
# ---------------------------------------------------------------------------


async def test_group_compact_independent_counter(prompt, short_term, tools, timeline) -> None:
    """Group compact failures should not affect private compact."""
    async for client in _client(prompt, short_term, tools, timeline=timeline, max_compact_failures=2):
        client._group_compact_failures = 2
        assert client._private_compact_failures == 0

        _fill_messages(short_term, "private_100")

        with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=MOCK_RESULT):
            await client._compact("private_100")
        assert client._private_compact_failures == 0


# ---------------------------------------------------------------------------
# Compact — memo extraction (private)
# ---------------------------------------------------------------------------


async def test_private_compact_extracts_memo(prompt, short_term, tools, memo_store) -> None:
    async for client in _client(prompt, short_term, tools, memo_store=memo_store):
        _fill_messages(short_term, "private_12345")

        response = json.dumps({"summary": "test summary", "memo": "用户A｜测试"})
        mock_result = {"text": response, "tool_uses": [], "input_tokens": 100}
        with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_result):
            await client._compact("private_12345")

        memo = memo_store.read("user_12345")
        assert memo is not None
        assert "用户A" in memo.body


async def test_private_compact_json_fallback(prompt, short_term, tools) -> None:
    async for client in _client(prompt, short_term, tools):
        _fill_messages(short_term, "private_100")

        mock_result = {"text": "plain text summary", "tool_uses": [], "input_tokens": 100}
        with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_result):
            await client._compact("private_100")
        assert short_term.get_summary("private_100") == "plain text summary"


# ---------------------------------------------------------------------------
# Compact — memo extraction (group)
# ---------------------------------------------------------------------------


async def test_group_compact_extracts_memos(prompt, short_term, tools, timeline, memo_store) -> None:
    async for client in _client(prompt, short_term, tools, timeline=timeline, memo_store=memo_store):
        gid = "99999"
        for i in range(8):
            timeline.add(gid, role="user", content=f"msg {i}", speaker=f"nick({i * 111})")

        response = json.dumps({
            "summary": "group summary",
            "memos": {"111": "用户1｜test", "222": "用户2｜test"},
            "group_memo": "群信息｜test",
        })
        mock_result = {"text": response, "tool_uses": [], "input_tokens": 100}

        with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_result):
            await client._compact_group(gid, _IDENTITY)

        assert memo_store.read("user_111") is not None
        assert memo_store.read("user_222") is not None
        assert memo_store.read("group_99999") is not None


async def test_group_compact_rejects_non_digit_uid(prompt, short_term, tools, timeline, memo_store) -> None:
    async for client in _client(prompt, short_term, tools, timeline=timeline, memo_store=memo_store):
        gid = "99999"
        for i in range(8):
            timeline.add(gid, role="user", content=f"msg {i}", speaker=f"nick({i})")

        response = json.dumps({
            "summary": "summary",
            "memos": {"../../etc/passwd": "evil", "valid_not_digit": "ignored", "12345": "good"},
        })
        mock_result = {"text": response, "tool_uses": [], "input_tokens": 100}

        with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_result):
            await client._compact_group(gid, _IDENTITY)

        assert memo_store.read("user_12345") is not None
        assert memo_store.read("user_../../etc/passwd") is None


# ---------------------------------------------------------------------------
# on_compact callback
# ---------------------------------------------------------------------------


async def test_compact_calls_on_compact(prompt, short_term, tools) -> None:
    async for client in _client(prompt, short_term, tools):
        callback = Mock()
        client._on_compact = callback
        _fill_messages(short_term, "private_100")

        with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=MOCK_RESULT):
            await client._compact("private_100")
        callback.assert_called_once()
