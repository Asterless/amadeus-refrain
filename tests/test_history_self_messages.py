"""Tests that bot's own messages are correctly included in history loading."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.group_timeline import GroupTimeline
from src.memory.history_loader import load_group_history


def _make_napcat_session(messages: list[dict[str, Any]]) -> MagicMock:
    """Build a mock aiohttp session that returns history messages."""
    payload = {"retcode": 0, "data": {"messages": messages}}

    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value=payload)
    mock_resp.status = 200

    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session_cls = MagicMock()
    mock_session_cls.__aenter__ = AsyncMock(return_value=MagicMock(post=MagicMock(return_value=mock_cm)))
    mock_session_cls.__aexit__ = AsyncMock(return_value=False)
    return mock_session_cls


def _msg(user_id: int, text: str, nickname: str = "user", msg_id: int = 1) -> dict[str, Any]:
    return {
        "sender": {"user_id": user_id, "nickname": nickname},
        "message": [{"type": "text", "data": {"text": text}}],
        "message_id": msg_id,
    }


@pytest.fixture
def timeline() -> GroupTimeline:
    return GroupTimeline()


async def test_bot_messages_classified_as_assistant(timeline: GroupTimeline, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bot's own messages should be role=assistant in the timeline."""
    bot_id = "123456"
    messages = [
        _msg(999, "hello", "Alice", 1),
        _msg(123456, "hi Alice!", "Bot", 2),  # bot's own message
        _msg(999, "how are you?", "Alice", 3),
    ]

    import src.memory.history_loader as hl

    original = hl._load_one_group

    async def patched_load(session: Any, napcat_url: str, group_id: str, tl: GroupTimeline,
                           count: int, bot_self_id: str = "", image_cache: Any = None,
                           sticker_store: Any = None) -> None:
        # Directly invoke with our mock session
        await original(session, napcat_url, group_id, tl, count, bot_self_id, image_cache, sticker_store)

    # Build a mock that returns our messages
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"retcode": 0, "data": {"messages": messages}})
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_cm)

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    import aiohttp
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: mock_session_ctx)

    await load_group_history(
        napcat_url="http://localhost:3000",
        group_ids=["100"],
        timeline=timeline,
        count=30,
        bot_self_id=bot_id,
    )

    msgs = timeline.get_messages("100")
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "hi Alice!"
    assert msgs[2]["role"] == "user"


async def test_bot_messages_produce_valid_anthropic_format(timeline: GroupTimeline) -> None:
    """Timeline with bot messages should produce valid alternating-role Anthropic messages."""
    timeline.add("100", role="user", speaker="Alice(999)", content="hello")
    timeline.add("100", role="assistant", content="hi!")
    timeline.add("100", role="user", speaker="Alice(999)", content="how are you?")

    result = timeline.to_anthropic_messages("100")
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "hi!"
    assert result[2]["role"] == "user"


async def test_consecutive_assistant_messages_merged(timeline: GroupTimeline) -> None:
    """Consecutive assistant messages (from history reload) should be merged."""
    timeline.add("100", role="user", speaker="Alice(999)", content="hello")
    timeline.add("100", role="assistant", content="hi!")
    timeline.add("100", role="assistant", content="how can I help?")
    timeline.add("100", role="user", speaker="Alice(999)", content="thanks")

    result = timeline.to_anthropic_messages("100")
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "hi!\nhow can I help?"
    assert result[2]["role"] == "user"


async def test_no_bot_self_id_all_user(timeline: GroupTimeline, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without bot_self_id, ALL messages become user role (the bug scenario)."""
    messages = [
        _msg(999, "hello", "Alice", 1),
        _msg(123456, "hi Alice!", "Bot", 2),
        _msg(999, "how are you?", "Alice", 3),
    ]

    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"retcode": 0, "data": {"messages": messages}})
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_cm)

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

    import aiohttp
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: mock_session_ctx)

    await load_group_history(
        napcat_url="http://localhost:3000",
        group_ids=["100"],
        timeline=timeline,
        count=30,
        bot_self_id="",  # no bot ID → all treated as user
    )

    msgs = timeline.get_messages("100")
    assert len(msgs) == 3
    # Without bot_self_id, bot's message is wrongly classified as user
    assert all(m["role"] == "user" for m in msgs)
