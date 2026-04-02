"""GroupTimeline 的单元测试。"""



from src.memory.group_timeline import GroupTimeline

# ---------------------------------------------------------------------------
# test_add_and_get_messages
# ---------------------------------------------------------------------------


def test_add_and_get_messages(group_timeline: GroupTimeline) -> None:
    group_timeline.add("g1", role="user", content="你好", speaker="Alice(123)")
    group_timeline.add("g1", role="assistant", content="你好！有什么可以帮你？")
    messages = group_timeline.get_messages("g1")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "你好"
    assert messages[0]["speaker"] == "Alice(123)"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "你好！有什么可以帮你？"
    assert messages[1]["speaker"] is None


# ---------------------------------------------------------------------------
# test_to_anthropic_merges_consecutive_users
# ---------------------------------------------------------------------------


def test_to_anthropic_merges_consecutive_users(group_timeline: GroupTimeline) -> None:
    group_timeline.add("g1", role="user", content="第一条", speaker="Alice(1)")
    group_timeline.add("g1", role="user", content="第二条", speaker="Bob(2)")
    group_timeline.add("g1", role="assistant", content="好的")
    group_timeline.add("g1", role="user", content="第三条", speaker="Alice(1)")

    msgs = group_timeline.to_anthropic_messages("g1")
    # 连续两条 user 消息合并为一块，接着 assistant，最后一条 user
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Alice(1): 第一条\nBob(2): 第二条"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "好的"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == "Alice(1): 第三条"


# ---------------------------------------------------------------------------
# test_to_anthropic_empty
# ---------------------------------------------------------------------------


def test_to_anthropic_empty(group_timeline: GroupTimeline) -> None:
    result = group_timeline.to_anthropic_messages("nonexistent")
    assert result == []


# ---------------------------------------------------------------------------
# test_group_isolation
# ---------------------------------------------------------------------------


def test_group_isolation(group_timeline: GroupTimeline) -> None:
    group_timeline.add("g1", role="user", content="群1消息", speaker="A(1)")
    group_timeline.add("g2", role="user", content="群2消息", speaker="B(2)")

    assert len(group_timeline.get_messages("g1")) == 1
    assert group_timeline.get_messages("g1")[0]["content"] == "群1消息"
    assert len(group_timeline.get_messages("g2")) == 1
    assert group_timeline.get_messages("g2")[0]["content"] == "群2消息"


# ---------------------------------------------------------------------------
# test_max_messages_eviction
# ---------------------------------------------------------------------------


def test_max_messages_eviction() -> None:
    tl = GroupTimeline(max_messages=3)
    tl.add("g1", role="user", content="msg1", speaker="A(1)")
    tl.add("g1", role="user", content="msg2", speaker="A(1)")
    tl.add("g1", role="user", content="msg3", speaker="A(1)")
    tl.add("g1", role="user", content="msg4", speaker="A(1)")

    msgs = tl.get_messages("g1")
    assert len(msgs) == 3
    assert msgs[0]["content"] == "msg2"
    assert msgs[-1]["content"] == "msg4"


# ---------------------------------------------------------------------------
# test_assistant_no_speaker
# ---------------------------------------------------------------------------


def test_assistant_no_speaker(group_timeline: GroupTimeline) -> None:
    group_timeline.add("g1", role="assistant", content="我是助手")
    msgs = group_timeline.get_messages("g1")
    assert msgs[0]["speaker"] is None

    anthropic = group_timeline.to_anthropic_messages("g1")
    assert anthropic[0]["role"] == "assistant"
    assert anthropic[0]["content"] == "我是助手"


# ---------------------------------------------------------------------------
# test_compact
# ---------------------------------------------------------------------------


def test_compact(group_timeline: GroupTimeline) -> None:
    for i in range(6):
        role = "assistant" if i % 2 else "user"
        group_timeline.add("g1", role=role, content=f"msg{i}", speaker="A(1)" if role == "user" else None)

    group_timeline.set_input_tokens("g1", 5000)
    assert group_timeline.get_input_tokens("g1") == 5000

    group_timeline.compact("g1", split=2, new_summary="前两条已压缩")

    msgs = group_timeline.get_messages("g1")
    assert len(msgs) == 4
    assert msgs[0]["content"] == "msg2"

    assert group_timeline.get_summary("g1") == "前两条已压缩"
    assert group_timeline.get_input_tokens("g1") == 0


# ---------------------------------------------------------------------------
# test_cached_msg_index
# ---------------------------------------------------------------------------


def test_cached_msg_index_default(group_timeline: GroupTimeline) -> None:
    """Default cached message index is 0."""
    assert group_timeline.get_cached_msg_index("g1") == 0


def test_cached_msg_index_set_and_get(group_timeline: GroupTimeline) -> None:
    group_timeline.set_cached_msg_index("g1", 5)
    assert group_timeline.get_cached_msg_index("g1") == 5


def test_cached_msg_index_reset_on_compact(group_timeline: GroupTimeline) -> None:
    """compact resets cached index so stale breakpoints don't persist."""
    for i in range(4):
        role = "assistant" if i % 2 else "user"
        group_timeline.add("g1", role=role, content=f"msg{i}", speaker="A(1)" if role == "user" else None)
    group_timeline.set_cached_msg_index("g1", 3)
    group_timeline.compact("g1", split=2, new_summary="摘要")
    assert group_timeline.get_cached_msg_index("g1") == 0


# ---------------------------------------------------------------------------
# test_drop_oldest
# ---------------------------------------------------------------------------


def test_drop_oldest(group_timeline: GroupTimeline) -> None:
    for i in range(10):
        group_timeline.add("g1", role="user", speaker=f"u{i}", content=f"msg{i}")
    assert len(group_timeline.get_messages("g1")) == 10
    group_timeline.drop_oldest("g1", 3)
    messages = group_timeline.get_messages("g1")
    assert len(messages) == 7
    assert messages[0]["content"] == "msg3"
