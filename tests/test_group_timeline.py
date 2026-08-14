"""GroupTimeline unit tests — _TurnLog + turns/pending model."""

import asyncio
import json

import aiosqlite
import pytest

from src.memory.group_timeline import GroupTimeline, _TurnLog
from src.memory.message_log import MessageLog

# ===========================================================================
# TestTurnLog
# ===========================================================================


class TestTurnLog:
    """Verify _TurnLog is an append-only Sequence[dict]."""

    def test_append_and_len(self) -> None:
        log = _TurnLog()
        assert len(log) == 0
        log.append({"role": "user", "content": "hi"})
        assert len(log) == 1
        log.append({"role": "assistant", "content": "hello"})
        assert len(log) == 2

    def test_getitem(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "a"})
        log.append({"role": "assistant", "content": "b"})
        assert log[0]["content"] == "a"
        assert log[1]["content"] == "b"
        assert log[-1]["content"] == "b"

    def test_getitem_slice(self) -> None:
        log = _TurnLog()
        for i in range(5):
            log.append({"role": "user", "content": f"m{i}"})
        sliced = log[1:3]
        assert len(sliced) == 2
        assert sliced[0]["content"] == "m1"
        assert sliced[1]["content"] == "m2"

    def test_bool_empty(self) -> None:
        log = _TurnLog()
        assert not log

    def test_bool_nonempty(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "x"})
        assert log

    def test_iter(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "a"})
        log.append({"role": "assistant", "content": "b"})
        items = list(log)
        assert len(items) == 2
        assert items[0]["content"] == "a"

    def test_no_setitem(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "x"})
        with pytest.raises(TypeError):
            log[0] = {"role": "user", "content": "y"}  # type: ignore[index]

    def test_no_delitem(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "x"})
        with pytest.raises(TypeError):
            del log[0]  # type: ignore[attr-defined]

    def test_compact_truncate(self) -> None:
        log = _TurnLog()
        for i in range(6):
            log.append({"role": "user", "content": f"m{i}"})
        log.compact_truncate(4)
        assert len(log) == 2
        assert log[0]["content"] == "m4"
        assert log[1]["content"] == "m5"

    def test_compact_truncate_zero(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "a"})
        log.compact_truncate(0)
        assert len(log) == 1

    def test_reset(self) -> None:
        log = _TurnLog()
        log.append({"role": "user", "content": "x"})
        log.append({"role": "assistant", "content": "y"})
        log.reset()
        assert len(log) == 0
        assert not log


# ===========================================================================
# TestGroupTimelineLifecycle
# ===========================================================================


class TestGroupTimelineLifecycle:
    """Verify the add-user/add-assistant flush model."""

    @pytest.fixture()
    def tl(self) -> GroupTimeline:
        return GroupTimeline()

    # -- add(role="user") goes to pending, not turns --

    def test_add_user_goes_to_pending(self, tl: GroupTimeline) -> None:
        tl.add("g1", role="user", content="hello", speaker="Alice(1)")
        assert len(tl.get_turns("g1")) == 0
        pending = tl.get_pending("g1")
        assert len(pending) == 1
        assert pending[0]["content"] == "hello"

    # -- add(role="assistant") flushes pending --

    def test_add_assistant_flushes_pending(self, tl: GroupTimeline) -> None:
        tl.add("g1", role="user", content="hi", speaker="Alice(1)")
        tl.add("g1", role="user", content="how are you?", speaker="Bob(2)")
        tl.add("g1", role="assistant", content="I'm fine!")

        turns = tl.get_turns("g1")
        assert len(turns) == 2  # 1 merged user + 1 assistant
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "I'm fine!"

        # pending should be empty after flush
        assert len(tl.get_pending("g1")) == 0

    def test_commit_assistant_preserves_messages_arriving_after_snapshot(self, tl: GroupTimeline) -> None:
        tl.add("g1", role="user", content="first", speaker="Alice(1)", message_id=10)
        snapshot_count = len(tl.get_pending("g1"))
        tl.add("g1", role="user", content="arrived later", speaker="Bob(2)", message_id=11)

        tl.commit_assistant("g1", "reply to first", pending_count=snapshot_count)

        turns = tl.get_turns("g1")
        assert len(turns) == 2
        assert "first" in str(turns[0]["content"])
        assert "arrived later" not in str(turns[0]["content"])
        pending = tl.get_pending("g1")
        assert [message.get("message_id") for message in pending] == [11]

    def test_pending_count_through_message_id(self, tl: GroupTimeline) -> None:
        for message_id in (10, 11, 12):
            tl.add("g1", role="user", content=str(message_id), message_id=message_id)

        assert tl.pending_count_through("g1", 11) == 2
        assert tl.pending_count_through("g1", None) == 3
        assert tl.pending_count_through("g1", 999) == 3

    # -- add(role="assistant") without pending produces assistant-only turn --

    def test_add_assistant_without_pending(self, tl: GroupTimeline) -> None:
        tl.add("g1", role="assistant", content="proactive hello")
        turns = tl.get_turns("g1")
        assert len(turns) == 1
        assert turns[0]["role"] == "assistant"
        assert turns[0]["content"] == "proactive hello"

    # -- multiple rounds --

    def test_multiple_rounds(self, tl: GroupTimeline) -> None:
        # Round 1
        tl.add("g1", role="user", content="Q1", speaker="Alice(1)")
        tl.add("g1", role="assistant", content="A1")
        # Round 2
        tl.add("g1", role="user", content="Q2", speaker="Bob(2)")
        tl.add("g1", role="assistant", content="A2")

        turns = tl.get_turns("g1")
        assert len(turns) == 4  # user, assistant, user, assistant
        assert turns[0]["role"] == "user"
        assert turns[1]["content"] == "A1"
        assert turns[2]["role"] == "user"
        assert turns[3]["content"] == "A2"

    # -- pending preserved across reads --

    def test_pending_preserved_across_reads(self, tl: GroupTimeline) -> None:
        tl.add("g1", role="user", content="msg1", speaker="A(1)")
        tl.add("g1", role="user", content="msg2", speaker="B(2)")
        p1 = tl.get_pending("g1")
        p2 = tl.get_pending("g1")
        assert len(p1) == 2
        assert len(p2) == 2
        # get_pending returns a copy, so mutating one doesn't affect the other
        p1.clear()
        assert len(tl.get_pending("g1")) == 2

    # -- group isolation --

    def test_group_isolation(self, tl: GroupTimeline) -> None:
        tl.add("g1", role="user", content="g1 msg", speaker="A(1)")
        tl.add("g2", role="user", content="g2 msg", speaker="B(2)")

        assert len(tl.get_pending("g1")) == 1
        assert len(tl.get_pending("g2")) == 1
        assert tl.get_pending("g1")[0]["content"] == "g1 msg"
        assert tl.get_pending("g2")[0]["content"] == "g2 msg"

        assert len(tl.get_turns("g1")) == 0
        assert len(tl.get_turns("g2")) == 0

    # -- turn_times_recorded --

    def test_turn_times_recorded(self, tl: GroupTimeline) -> None:
        tl.add("g1", role="user", content="Q", speaker="A(1)")
        tl.add("g1", role="assistant", content="A")

        turns = tl.get_turns("g1")
        assert len(turns) == 2
        # Both user and assistant turns should have timestamps
        t0 = tl.get_turn_time("g1", 0)
        t1 = tl.get_turn_time("g1", 1)
        assert t0 > 0
        assert t1 >= t0

    # -- clear preserves summary --

    def test_clear_preserves_summary(self, tl: GroupTimeline) -> None:
        tl.add("g1", role="user", content="Q", speaker="A(1)")
        tl.add("g1", role="assistant", content="A")
        tl.compact("g1", split=0, new_summary="some summary")

        tl.clear("g1")
        assert len(tl.get_turns("g1")) == 0
        assert len(tl.get_pending("g1")) == 0
        assert tl.get_summary("g1") == "some summary"

    # -- compact --

    def test_compact(self, tl: GroupTimeline) -> None:
        for i in range(3):
            tl.add("g1", role="user", content=f"Q{i}", speaker="A(1)")
            tl.add("g1", role="assistant", content=f"A{i}")
        tl.set_input_tokens("g1", 5000)

        # 6 turns: [user:Q0, asst:A0, user:Q1, asst:A1, user:Q2, asst:A2]
        # split=2 removes first 2 → [user:Q1, asst:A1, user:Q2, asst:A2]
        tl.compact("g1", split=2, new_summary="compressed")

        turns = tl.get_turns("g1")
        assert len(turns) == 4  # 6 - 2 = 4
        assert turns[0]["role"] == "user"
        assert "Q1" in str(turns[0]["content"])
        assert turns[1]["content"] == "A1"
        assert tl.get_summary("g1") == "compressed"
        assert tl.get_input_tokens("g1") == 0

    # -- drop_oldest --

    def test_drop_oldest(self, tl: GroupTimeline) -> None:
        for i in range(3):
            tl.add("g1", role="user", content=f"Q{i}", speaker="A(1)")
            tl.add("g1", role="assistant", content=f"A{i}")

        # 6 turns, drop first 2 → [user:Q1, asst:A1, user:Q2, asst:A2]
        tl.drop_oldest("g1", 2)
        turns = tl.get_turns("g1")
        assert len(turns) == 4
        assert turns[0]["role"] == "user"
        assert "Q1" in str(turns[0]["content"])

    # -- get_turns returns read-only Sequence --

    def test_get_turns_is_sequence(self, tl: GroupTimeline) -> None:
        tl.add("g1", role="user", content="Q", speaker="A(1)")
        tl.add("g1", role="assistant", content="A")

        turns = tl.get_turns("g1")
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        # It's a _TurnLog — setitem should fail
        with pytest.raises(TypeError):
            turns[0] = {"role": "user", "content": "hacked"}  # type: ignore[index]

    # -- merged user content --

    def test_merged_user_content(self, tl: GroupTimeline) -> None:
        """Multiple user messages before an assistant turn are merged."""
        tl.add("g1", role="user", content="first", speaker="Alice(1)")
        tl.add("g1", role="user", content="second", speaker="Bob(2)")
        tl.add("g1", role="assistant", content="reply")

        turns = tl.get_turns("g1")
        assert turns[0]["role"] == "user"
        # Merged content should contain both speakers
        content = turns[0]["content"]
        assert isinstance(content, str)
        assert "Alice(1): first" in content
        assert "Bob(2): second" in content

    # -- needs_compact --

    def test_needs_compact(self, tl: GroupTimeline) -> None:
        tl.set_input_tokens("g1", 8000)
        assert tl.needs_compact("g1", max_tokens=10000, ratio=0.7)
        assert not tl.needs_compact("g1", max_tokens=10000, ratio=0.9)

    # -- cached_msg_index --

    def test_cached_msg_index_default(self, tl: GroupTimeline) -> None:
        assert tl.get_cached_msg_index("g1") == 0

    def test_cached_msg_index_set_get(self, tl: GroupTimeline) -> None:
        tl.set_cached_msg_index("g1", 5)
        assert tl.get_cached_msg_index("g1") == 5

    def test_cached_msg_index_reset_on_compact(self, tl: GroupTimeline) -> None:
        for i in range(3):
            tl.add("g1", role="user", content=f"Q{i}", speaker="A(1)")
            tl.add("g1", role="assistant", content=f"A{i}")
        tl.set_cached_msg_index("g1", 3)
        tl.compact("g1", split=2, new_summary="summary")
        assert tl.get_cached_msg_index("g1") == 0


# ===========================================================================
# TestMessageLogIntegration
# ===========================================================================


class TestMessageLogIntegration:
    """Verify GroupTimeline.add() fires-and-forgets SQLite writes via MessageLog."""

    @pytest.mark.asyncio()
    async def test_add_writes_to_message_log(self, tmp_path: object) -> None:
        db_path = f"{tmp_path}/messages.db"
        ml = MessageLog(db_path=db_path)
        await ml.init()

        tl = GroupTimeline(message_log=ml)
        tl.add("g1", role="user", content="hello world", speaker="Alice(111)")
        tl.add("g1", role="assistant", content="hi there")

        # Let fire-and-forget tasks complete
        await asyncio.sleep(0.1)

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT group_id, role, speaker, content_text, content_json, message_id "
                "FROM group_messages ORDER BY id"
            )
            rows = [dict(r) for r in await cursor.fetchall()]

        assert len(rows) == 2

        # User message
        assert rows[0]["group_id"] == "g1"
        assert rows[0]["role"] == "user"
        assert rows[0]["speaker"] == "Alice(111)"
        assert rows[0]["content_text"] == "hello world"
        assert json.loads(rows[0]["content_json"]) == "hello world"
        assert rows[0]["message_id"] is None

        # Assistant message
        assert rows[1]["role"] == "assistant"
        assert rows[1]["speaker"] is None
        assert rows[1]["content_text"] == "hi there"

        await ml.close()

    @pytest.mark.asyncio()
    async def test_add_multimodal_writes_both_columns(self, tmp_path: object) -> None:
        db_path = f"{tmp_path}/messages.db"
        ml = MessageLog(db_path=db_path)
        await ml.init()

        tl = GroupTimeline(message_log=ml)
        multimodal_content = [
            {"type": "text", "text": "look at this"},
            {"type": "image_ref", "path": "storage/img/abc.jpg", "media_type": "image/jpeg"},
        ]
        tl.add(
            "g2",
            role="user",
            content=multimodal_content,  # type: ignore[arg-type]
            speaker="Bob(222)",
            message_id=42,
        )

        await asyncio.sleep(0.1)

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT content_text, content_json, message_id FROM group_messages"
            )
            row = dict(await cursor.fetchone())  # type: ignore[arg-type]

        # content_text has only the text portions
        assert row["content_text"] == "look at this"  # type: ignore[index]

        # content_json has the full block array including image_ref
        parsed = json.loads(row["content_json"])  # type: ignore[index]
        assert len(parsed) == 2
        assert parsed[0]["type"] == "text"
        assert parsed[1]["type"] == "image_ref"
        assert parsed[1]["path"] == "storage/img/abc.jpg"

        assert row["message_id"] == 42  # type: ignore[index]

        await ml.close()
