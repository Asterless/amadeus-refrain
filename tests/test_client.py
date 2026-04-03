"""Tests for LLMClient compact logic, circuit breaker, and micro compact."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.identity.models import Identity
from src.llm.client import LLMClient, _content_text, _resolve_image_refs, _to_anthropic_message, _ToolUse
from src.llm.prompt import PromptBuilder
from src.llm.usage import UsageTracker
from src.memory.group_timeline import GroupTimeline
from src.memory.memo_store import MemoStore
from src.memory.short_term import ChatMessage, ShortTermMemory
from src.memory.types import ContentBlock, ImageRefBlock, TextBlock
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
    return GroupTimeline()


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
        compact_ratio=0.7,
        compress_ratio=0.5,
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


MOCK_RESULT = {
    "text": "summary", "tool_uses": [], "input_tokens": 100,
    "output_tokens": 50, "cache_read": 0, "cache_create": 0,
}

MOCK_RESULT_FULL = {
    "text": "reply text",
    "tool_uses": [],
    "input_tokens": 160,  # total = 100 + 50 + 10
    "output_tokens": 200,
    "cache_read": 50,
    "cache_create": 10,
}


# ---------------------------------------------------------------------------
# Compact trigger — single ratio
# ---------------------------------------------------------------------------


async def test_group_compact_triggers_at_ratio(prompt, short_term, tools, timeline, memo_store) -> None:
    """compact_group fires when input_tokens > max_context_tokens * compact_ratio."""
    async for client in _client(prompt, short_term, tools, timeline=timeline, memo_store=memo_store):
        gid = "12345"
        for i in range(8):
            timeline.add(gid, role="user", content=f"msg {i}", speaker=f"user({i})")

        # Simulate previous call reported tokens above threshold (70k > 100k * 0.7)
        timeline.set_input_tokens(gid, 70_001)

        mock_compact = {"text": "compressed", "tool_uses": [], "input_tokens": 50}
        mock_chat = {
            "text": "reply", "tool_uses": [], "input_tokens": 5000,
            "output_tokens": 100, "cache_read": 0, "cache_create": 0,
        }

        with patch("src.llm.client._call_api", new_callable=AsyncMock, side_effect=[mock_compact, mock_chat]):
            result = await client.chat(
                session_id="group_12345", user_id="111",
                user_content="hello", identity=_IDENTITY, group_id=gid,
            )

        assert result is not None
        # After compact, 4 messages remain (8 * 0.5), plus 1 assistant reply added by chat()
        assert len(timeline.get_messages(gid)) == 5  # 8 * 0.5 = 4 remaining + 1 assistant reply
        assert timeline.get_summary(gid) == "compressed"


async def test_group_no_compact_below_ratio(prompt, short_term, tools, timeline, memo_store) -> None:
    """No compact when tokens below threshold."""
    async for client in _client(prompt, short_term, tools, timeline=timeline, memo_store=memo_store):
        gid = "12345"
        for i in range(8):
            timeline.add(gid, role="user", content=f"msg {i}", speaker=f"user({i})")
        timeline.set_input_tokens(gid, 50_000)  # below 70k threshold

        mock_chat = {
            "text": "reply", "tool_uses": [], "input_tokens": 5000,
            "output_tokens": 100, "cache_read": 0, "cache_create": 0,
        }
        with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_chat):
            await client.chat(
                session_id="group_12345", user_id="111",
                user_content="hello", identity=_IDENTITY, group_id=gid,
            )
        assert len(timeline.get_messages(gid)) == 9  # 8 original + 1 assistant reply added by chat()


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


async def test_private_compact_appends_memo(prompt, short_term, tools, memo_store) -> None:
    async for client in _client(prompt, short_term, tools, memo_store=memo_store):
        _fill_messages(short_term, "private_12345")

        # First call: LLM returns a tool call to append memo
        mock_tool_call = {
            "text": "",
            "tool_uses": [_ToolUse(id="tc1", name="append_memo", input={"id": "user_12345", "note": "用户A喜欢编程"})],
            "input_tokens": 100,
        }
        # Second call: LLM returns the summary text
        mock_summary = {"text": "test summary", "tool_uses": [], "input_tokens": 50}

        with patch("src.llm.client._call_api", new_callable=AsyncMock, side_effect=[mock_tool_call, mock_summary]):
            await client._compact("private_12345")

        memo = memo_store.read("user_12345")
        assert memo is not None
        assert "用户A喜欢编程" in memo.body
        assert short_term.get_summary("private_12345") == "test summary"


async def test_private_compact_no_memo_tools(prompt, short_term, tools) -> None:
    """Compact without memo store: LLM returns plain text summary, no tool calls."""
    async for client in _client(prompt, short_term, tools):
        _fill_messages(short_term, "private_100")

        mock_result = {"text": "plain text summary", "tool_uses": [], "input_tokens": 100}
        with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_result):
            await client._compact("private_100")
        assert short_term.get_summary("private_100") == "plain text summary"


# ---------------------------------------------------------------------------
# Compact — memo extraction (group)
# ---------------------------------------------------------------------------


async def test_group_compact_appends_memos(prompt, short_term, tools, timeline, memo_store) -> None:
    async for client in _client(prompt, short_term, tools, timeline=timeline, memo_store=memo_store):
        gid = "99999"
        for i in range(8):
            timeline.add(gid, role="user", content=f"msg {i}", speaker=f"nick({i * 111})")

        # First call: LLM returns tool calls to append memos
        mock_tool_call = {
            "text": "",
            "tool_uses": [
                _ToolUse(id="tc1", name="append_memo", input={"id": "user_111", "note": "用户1是前端开发"}),
                _ToolUse(id="tc2", name="append_memo", input={"id": "user_222", "note": "用户2喜欢Rust"}),
                _ToolUse(id="tc3", name="append_memo", input={"id": "group_99999", "note": "技术讨论群"}),
            ],
            "input_tokens": 100,
        }
        # Second call: LLM returns the summary
        mock_summary = {"text": "group summary", "tool_uses": [], "input_tokens": 50}

        with patch("src.llm.client._call_api", new_callable=AsyncMock, side_effect=[mock_tool_call, mock_summary]):
            await client._compact_group(gid, _IDENTITY)

        assert memo_store.read("user_111") is not None
        assert memo_store.read("user_222") is not None
        assert memo_store.read("group_99999") is not None
        assert timeline.get_summary(gid) == "group summary"


async def test_group_compact_rejects_path_traversal(prompt, short_term, tools, timeline, memo_store) -> None:
    """Path traversal IDs are rejected gracefully, valid IDs still succeed."""
    async for client in _client(prompt, short_term, tools, timeline=timeline, memo_store=memo_store):
        gid = "99999"
        for i in range(8):
            timeline.add(gid, role="user", content=f"msg {i}", speaker=f"nick({i})")

        # LLM tries to write to a path traversal ID and a valid ID
        mock_tool_call = {
            "text": "",
            "tool_uses": [
                _ToolUse(id="tc1", name="append_memo", input={"id": "user_../../etc/passwd", "note": "evil"}),
                _ToolUse(id="tc2", name="append_memo", input={"id": "user_12345", "note": "good"}),
            ],
            "input_tokens": 100,
        }
        mock_summary = {"text": "summary", "tool_uses": [], "input_tokens": 50}

        with patch("src.llm.client._call_api", new_callable=AsyncMock, side_effect=[mock_tool_call, mock_summary]):
            await client._compact_group(gid, _IDENTITY)

        # Path traversal was rejected, valid write succeeded
        assert memo_store.read("user_12345") is not None
        assert timeline.get_summary(gid) == "summary"


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


# ---------------------------------------------------------------------------
# pass_turn behavior
# ---------------------------------------------------------------------------


class TestPassTurn:
    async def test_pass_turn_returns_none(self, prompt, short_term, tools, timeline, memo_store) -> None:
        """pass_turn is always honored — chat() returns None."""
        async for client in _client(prompt, short_term, tools, timeline=timeline, memo_store=memo_store):
            gid = "12345"
            timeline.add(gid, role="user", content="hello", speaker="user(111)")

            mock_result = {
                "text": "",
                "tool_uses": [_ToolUse(id="tu_1", name="pass_turn", input={"reason": "not relevant"})],
                "input_tokens": 100,
                "output_tokens": 0,
                "cache_read": 0,
                "cache_create": 0,
            }
            with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_result):
                result = await client.chat(
                    session_id="group_12345",
                    user_id="111",
                    user_content="hello",
                    identity=_IDENTITY,
                    group_id=gid,
                    ctx=None,
                )
            assert result is None


# ---------------------------------------------------------------------------
# UsageTracker integration
# ---------------------------------------------------------------------------


async def test_chat_records_usage(prompt, short_term, tools, tmp_path) -> None:
    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    await tracker.init()
    try:
        async for client in _client(prompt, short_term, tools):
            client._usage_tracker = tracker
            with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=MOCK_RESULT_FULL):
                await client.chat(
                    session_id="private_100", user_id="100",
                    user_content="hello", identity=_IDENTITY,
                )
            await asyncio.sleep(0)
        rows = await tracker.query_raw("SELECT * FROM llm_calls")
        assert len(rows) == 1
        row = rows[0]
        assert row["call_type"] == "chat"
        assert row["user_id"] == "100"
        assert row["output_tokens"] == 200
        assert row["input_tokens"] == 100  # raw = total(160) - cache_read(50) - cache_create(10)
        assert row["cache_read_tokens"] == 50
    finally:
        await tracker.close()


async def test_compact_records_usage(prompt, short_term, tools, tmp_path) -> None:
    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    await tracker.init()
    mock_result = {**MOCK_RESULT, "output_tokens": 80, "cache_read": 0, "cache_create": 0}
    try:
        async for client in _client(prompt, short_term, tools):
            client._usage_tracker = tracker
            _fill_messages(short_term, "private_100")
            with patch("src.llm.client._call_api", new_callable=AsyncMock, return_value=mock_result):
                await client._compact("private_100")
            await asyncio.sleep(0)
        rows = await tracker.query_raw("SELECT * FROM llm_calls WHERE call_type='compact'")
        assert len(rows) == 1
    finally:
        await tracker.close()


# ---------------------------------------------------------------------------
# _to_anthropic_message — content block passthrough
# ---------------------------------------------------------------------------


def test_to_anthropic_message_str() -> None:
    """String content passes through unchanged."""
    msg = ChatMessage(role="user", content="hello")
    result = _to_anthropic_message(msg)
    assert result == {"role": "user", "content": "hello"}


def test_to_anthropic_message_blocks() -> None:
    """Block content passes through as-is (image_ref converted later)."""
    blocks: list[ContentBlock] = [
        TextBlock(type="text", text="look"),
        ImageRefBlock(type="image_ref", path="cache/ab/abc.jpg", media_type="image/jpeg"),
    ]
    msg = ChatMessage(role="user", content=blocks)
    result = _to_anthropic_message(msg)
    assert result["role"] == "user"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2


# ---------------------------------------------------------------------------
# _content_text
# ---------------------------------------------------------------------------


def test_content_text_str() -> None:
    assert _content_text("hello") == "hello"


def test_content_text_blocks() -> None:
    blocks: list[ContentBlock] = [
        TextBlock(type="text", text="看这个"),
        ImageRefBlock(type="image_ref", path="x.jpg", media_type="image/jpeg"),
        TextBlock(type="text", text="不错吧"),
    ]
    assert _content_text(blocks) == "看这个 不错吧"


def test_content_text_empty_list() -> None:
    assert _content_text([]) == ""


# ---------------------------------------------------------------------------
# _resolve_image_refs
# ---------------------------------------------------------------------------


async def test_resolve_image_refs_none_cache() -> None:
    """With no image_cache, messages pass through unchanged."""
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_ref", "path": "x.jpg", "media_type": "image/jpeg"},
    ]}]
    result = await _resolve_image_refs(msgs, None)
    assert result[0]["content"][1]["type"] == "image_ref"  # unchanged


async def test_resolve_image_refs_resolves_to_base64() -> None:
    """image_ref blocks should be converted to base64 image blocks."""
    mock_cache = AsyncMock()
    mock_cache.load_as_base64.return_value = {
        "type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc123"},
    }
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "看"},
        {"type": "image_ref", "path": "x.jpg", "media_type": "image/jpeg"},
    ]}]
    result = await _resolve_image_refs(msgs, mock_cache)
    assert result[0]["content"][1]["type"] == "image"
    assert result[0]["content"][1]["source"]["data"] == "abc123"


async def test_resolve_image_refs_expired() -> None:
    """Expired images (load returns None) become [图片已过期]."""
    mock_cache = AsyncMock()
    mock_cache.load_as_base64.return_value = None
    msgs = [{"role": "user", "content": [
        {"type": "image_ref", "path": "gone.jpg", "media_type": "image/jpeg"},
    ]}]
    result = await _resolve_image_refs(msgs, mock_cache)
    assert result[0]["content"][0]["type"] == "text"
    assert "过期" in result[0]["content"][0]["text"]


async def test_resolve_image_refs_preserves_cache_control() -> None:
    """cache_control on image_ref blocks should transfer to resolved blocks."""
    mock_cache = AsyncMock()
    mock_cache.load_as_base64.return_value = {
        "type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "x"},
    }
    msgs = [{"role": "user", "content": [
        {"type": "image_ref", "path": "x.jpg", "media_type": "image/jpeg",
         "cache_control": {"type": "ephemeral"}},
    ]}]
    result = await _resolve_image_refs(msgs, mock_cache)
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
