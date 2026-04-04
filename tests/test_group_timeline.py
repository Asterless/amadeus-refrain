"""GroupTimeline unit tests — _TurnLog + turns/pending model."""

import pytest

from src.memory.group_timeline import GroupTimeline, _TurnLog

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
