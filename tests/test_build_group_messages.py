"""Tests for _build_group_messages: turns + pending merge + cache stability."""

from __future__ import annotations

from src.memory.group_timeline import GroupTimeline, _merge_user_contents


class TestBuildGroupMessages:
    """Exercise GroupTimeline's turns/pending model for _build_group_messages."""

    def test_turns_are_included_directly(self) -> None:
        """Turns from timeline are extended directly into messages."""
        tl = GroupTimeline()
        tl.add("g1", role="user", content="hello", speaker="Alice(1)")
        tl.add("g1", role="assistant", content="hi!")

        turns = tl.get_turns("g1")
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "hi!"

    def test_pending_merged_as_tail(self) -> None:
        """Pending user messages are merged as the last message."""
        tl = GroupTimeline()
        tl.add("g1", role="user", content="hello", speaker="Alice(1)")
        tl.add("g1", role="assistant", content="hi!")
        tl.add("g1", role="user", content="follow-up 1", speaker="Alice(1)")
        tl.add("g1", role="user", content="follow-up 2", speaker="Bob(2)")

        turns = tl.get_turns("g1")
        pending = tl.get_pending("g1")
        assert len(turns) == 2  # user + assistant
        assert len(pending) == 2  # two pending user messages

        # Merge pending like _build_group_messages does
        merged_content = _merge_user_contents(pending)
        assert isinstance(merged_content, str)
        assert "Alice(1): follow-up 1" in merged_content
        assert "Bob(2): follow-up 2" in merged_content

    def test_prefix_stability_across_appends(self) -> None:
        """Core cache invariant: turns prefix doesn't change when pending grows."""
        tl = GroupTimeline()
        tl.add("g1", role="user", content="Q1", speaker="Alice(1)")
        tl.add("g1", role="assistant", content="A1")

        # Snapshot turns before adding pending
        turns_before = list(tl.get_turns("g1"))

        # Add pending messages
        tl.add("g1", role="user", content="Q2a", speaker="Alice(1)")
        tl.add("g1", role="user", content="Q2b", speaker="Bob(2)")

        # Turns must be byte-identical
        turns_after = list(tl.get_turns("g1"))
        assert len(turns_before) == len(turns_after)
        for before, after in zip(turns_before, turns_after, strict=True):
            assert before is after  # same object identity

    def test_prefix_stability_after_flush(self) -> None:
        """After flush, old turns are the exact same objects (identity check)."""
        tl = GroupTimeline()
        tl.add("g1", role="user", content="Q1", speaker="Alice(1)")
        tl.add("g1", role="assistant", content="A1")

        # Capture references to existing turns
        turn0 = tl.get_turns("g1")[0]
        turn1 = tl.get_turns("g1")[1]

        # Add more messages and flush (assistant triggers flush)
        tl.add("g1", role="user", content="Q2", speaker="Bob(2)")
        tl.add("g1", role="assistant", content="A2")

        turns = tl.get_turns("g1")
        assert len(turns) == 4
        # Old turns remain the same objects
        assert turns[0] is turn0
        assert turns[1] is turn1

    def test_empty_group_returns_empty(self) -> None:
        """Non-existent group produces empty turns and pending."""
        tl = GroupTimeline()
        assert len(tl.get_turns("nonexistent")) == 0
        assert len(tl.get_pending("nonexistent")) == 0

    def test_pending_only_produces_single_message(self) -> None:
        """When there are only pending messages (no turns), merge produces one entry."""
        tl = GroupTimeline()
        tl.add("g1", role="user", content="msg1", speaker="Alice(1)")
        tl.add("g1", role="user", content="msg2", speaker="Bob(2)")

        turns = tl.get_turns("g1")
        pending = tl.get_pending("g1")
        assert len(turns) == 0
        assert len(pending) == 2

        merged = _merge_user_contents(pending)
        assert isinstance(merged, str)
        assert "Alice(1): msg1" in merged
        assert "Bob(2): msg2" in merged

    def test_summary_precedes_turns(self) -> None:
        """Summary should be present in the timeline state for building messages."""
        tl = GroupTimeline()
        tl.add("g1", role="user", content="Q", speaker="A(1)")
        tl.add("g1", role="assistant", content="A")
        tl.compact("g1", split=0, new_summary="this is a summary")

        assert tl.get_summary("g1") == "this is a summary"
        turns = tl.get_turns("g1")
        assert len(turns) == 2  # split=0 means no truncation
