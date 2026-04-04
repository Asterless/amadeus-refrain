"""Tests for MessageLog SQLite store."""

from __future__ import annotations

import json
import time

import pytest

from src.memory.message_log import MessageLog


@pytest.fixture
async def log(tmp_path):
    ml = MessageLog(db_path=str(tmp_path / "messages.db"))
    await ml.init()
    yield ml
    await ml.close()


async def test_record_and_query(log: MessageLog) -> None:
    """Insert 2 records, query, verify fields."""
    await log.record(
        group_id="111",
        role="user",
        speaker="Alice(12345)",
        content_text="hello",
        content_json=None,
        message_id=1001,
    )
    await log.record(
        group_id="111",
        role="assistant",
        speaker=None,
        content_text="hi there",
        content_json=None,
        message_id=1002,
    )
    rows = await log.query_for_compact("111", before=time.time() + 10)
    assert len(rows) == 2
    assert rows[0]["role"] == "user"
    assert rows[0]["speaker"] == "Alice(12345)"
    assert rows[0]["content_text"] == "hello"
    assert rows[0]["message_id"] == 1001
    assert rows[1]["role"] == "assistant"
    assert rows[1]["content_text"] == "hi there"
    assert rows[1]["speaker"] is None


async def test_query_respects_time_bound(log: MessageLog) -> None:
    """Insert 2 records with a time gap, query with cutoff, verify only older returned."""
    await log.record(
        group_id="111",
        role="user",
        speaker="A(1)",
        content_text="old",
        content_json=None,
    )
    # Query all to get the created_at of the first record
    all_rows = await log.query_for_compact("111", before=time.time() + 100)
    cutoff = all_rows[0]["created_at"] + 0.001

    # Insert a second record with created_at guaranteed to be after cutoff
    # We need to wait a tiny bit so the timestamp is strictly after cutoff
    # Instead, we'll use a cutoff between the two inserts
    await log.record(
        group_id="111",
        role="user",
        speaker="B(2)",
        content_text="new",
        content_json=None,
    )

    # Query with cutoff that only includes the first record
    rows = await log.query_for_compact("111", before=cutoff)
    assert len(rows) == 1
    assert rows[0]["content_text"] == "old"


async def test_group_isolation(log: MessageLog) -> None:
    """Insert in 2 groups, query one, verify isolation."""
    await log.record(
        group_id="aaa",
        role="user",
        speaker="X(1)",
        content_text="group a msg",
        content_json=None,
    )
    await log.record(
        group_id="bbb",
        role="user",
        speaker="Y(2)",
        content_text="group b msg",
        content_json=None,
    )
    rows_a = await log.query_for_compact("aaa", before=time.time() + 10)
    rows_b = await log.query_for_compact("bbb", before=time.time() + 10)
    assert len(rows_a) == 1
    assert rows_a[0]["content_text"] == "group a msg"
    assert len(rows_b) == 1
    assert rows_b[0]["content_text"] == "group b msg"


async def test_content_json_with_blocks(log: MessageLog) -> None:
    """Insert JSON with image_ref blocks, verify round-trip."""
    blocks = [
        {"type": "text", "text": "look at this"},
        {"type": "image_ref", "source": {"url": "https://example.com/img.png"}},
    ]
    json_str = json.dumps(blocks)
    await log.record(
        group_id="111",
        role="user",
        speaker="A(1)",
        content_text="look at this",
        content_json=json_str,
    )
    rows = await log.query_for_compact("111", before=time.time() + 10)
    assert len(rows) == 1
    assert rows[0]["content_json"] == json_str
    parsed = json.loads(rows[0]["content_json"])
    assert parsed == blocks
    assert parsed[1]["type"] == "image_ref"
