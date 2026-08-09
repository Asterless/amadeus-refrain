"""Tests for mandatory source links in meme-origin answers."""

from src.llm.client import (
    _ensure_source_link,
    _extract_source_urls,
    _latest_user_asks_for_source,
)


def test_detects_source_question_in_latest_user_message() -> None:
    messages = [
        {"role": "user", "content": "前文"},
        {"role": "assistant", "content": "回复"},
        {"role": "user", "content": [{"type": "text", "text": "丑橘这个梗出处在哪"}]},
    ]
    assert _latest_user_asks_for_source(messages) is True


def test_appends_preferred_platform_url() -> None:
    urls = ["https://example.com/mirror", "https://www.bilibili.com/video/BV123"]
    reply = _ensure_source_link("它是一只走红的橘猫。", True, urls)
    assert reply.endswith("来源：https://www.bilibili.com/video/BV123")


def test_does_not_duplicate_existing_url() -> None:
    reply = "来源：https://example.com/original"
    assert _ensure_source_link(reply, True, ["https://other.example"]) == reply


def test_marks_missing_verifiable_link() -> None:
    reply = _ensure_source_link("暂时无法确认来源。", True, [])
    assert reply.endswith("来源：暂未找到可验证链接")


def test_extract_source_urls_strips_sentence_punctuation() -> None:
    result = _extract_source_urls("来源 https://example.com/a。另见 https://example.com/b)")
    assert result == ["https://example.com/a", "https://example.com/b"]
