"""Tests for UsageTracker."""

import asyncio

import pytest

from src.llm.usage import UsageTracker


@pytest.fixture
async def tracker(tmp_path) -> UsageTracker:
    t = UsageTracker(db_path=str(tmp_path / "usage.db"))
    await t.init()
    return t


async def test_init_creates_table(tracker: UsageTracker) -> None:
    """Table should exist after init."""
    rows = await tracker.query_raw("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_calls'")
    assert len(rows) == 1


async def test_record_inserts_row(tracker: UsageTracker) -> None:
    await tracker.record(
        call_type="chat",
        user_id="12345",
        group_id=None,
        model="claude-sonnet-4-6",
        input_tokens=100,
        cache_read_tokens=50,
        cache_create_tokens=10,
        output_tokens=200,
        tool_rounds=0,
        elapsed_s=1.5,
    )
    rows = await tracker.query_raw("SELECT * FROM llm_calls")
    assert len(rows) == 1
    row = rows[0]
    assert row["call_type"] == "chat"
    assert row["user_id"] == "12345"
    assert row["group_id"] is None
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 200
    assert row["cache_read_tokens"] == 50
    assert row["cache_create_tokens"] == 10
    assert row["tool_rounds"] == 0
    assert row["elapsed_s"] == pytest.approx(1.5)
    assert row["error"] is None


async def test_record_with_error(tracker: UsageTracker) -> None:
    await tracker.record(
        call_type="chat",
        user_id="12345",
        group_id="99999",
        model="claude-sonnet-4-6",
        input_tokens=0,
        cache_read_tokens=0,
        cache_create_tokens=0,
        output_tokens=0,
        tool_rounds=0,
        elapsed_s=0.5,
        error="API timeout",
    )
    rows = await tracker.query_raw("SELECT error, group_id FROM llm_calls")
    assert rows[0]["error"] == "API timeout"
    assert rows[0]["group_id"] == "99999"


async def test_record_failure_does_not_raise(tracker: UsageTracker, tmp_path) -> None:
    """record() should swallow errors gracefully."""
    await tracker.close()
    # After close, writing should not raise
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=0, cache_read_tokens=0, cache_create_tokens=0,
        output_tokens=0, tool_rounds=0, elapsed_s=0.0,
    )
