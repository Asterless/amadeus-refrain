"""Tests for _call_api SSE parsing."""

import json
from unittest.mock import AsyncMock, MagicMock

from src.llm.client import _call_api


def _make_sse_lines(*events: str) -> list[bytes]:
    """Build raw SSE byte lines from event JSON strings."""
    lines: list[bytes] = []
    for event in events:
        lines.append(f"data: {event}\n".encode())
    return lines


def _mock_session(sse_lines: list[bytes]) -> MagicMock:
    """Create a mock aiohttp session whose post() yields given SSE lines."""

    async def _aiter():
        for line in sse_lines:
            yield line

    resp = AsyncMock()
    resp.raise_for_status = MagicMock()
    resp.content.__aiter__ = lambda self: _aiter().__aiter__()

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post.return_value = ctx
    return session


async def test_output_tokens_from_message_delta() -> None:
    """output_tokens should come from the last message_delta, not message_start."""
    events = [
        json.dumps({
            "type": "message_start",
            "message": {
                "usage": {"input_tokens": 100, "output_tokens": 1,
                           "cache_read_input_tokens": 50, "cache_creation_input_tokens": 10}
            },
        }),
        json.dumps({"type": "content_block_start", "content_block": {"type": "text", "text": ""}}),
        json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}),
        json.dumps({"type": "content_block_stop"}),
        json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 42}}),
    ]

    session = _mock_session(_make_sse_lines(*events))
    result = await _call_api(session, "http://fake", "sk-test", "model", [], [{"role": "user", "content": "hi"}])

    assert result["output_tokens"] == 42
    assert result["text"] == "Hello"
    assert result["input_tokens"] == 100 + 50 + 10  # total input
    assert result["cache_read"] == 50
    assert result["cache_create"] == 10
