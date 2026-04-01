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
# test_should_warm_basic
# ---------------------------------------------------------------------------


def test_should_warm_basic(group_timeline: GroupTimeline) -> None:
    group_timeline.set_input_tokens("g1", 1000)
    for i in range(10):
        group_timeline.add("g1", role="user", content=f"msg{i}", speaker="A(1)")

    assert group_timeline.should_warm("g1", interval=10, ttl=3600) is True


# ---------------------------------------------------------------------------
# test_should_warm_not_enough_messages
# ---------------------------------------------------------------------------


def test_should_warm_not_enough_messages(group_timeline: GroupTimeline) -> None:
    group_timeline.set_input_tokens("g1", 1000)
    for i in range(5):
        group_timeline.add("g1", role="user", content=f"msg{i}", speaker="A(1)")

    assert group_timeline.should_warm("g1", interval=10, ttl=3600) is False


# ---------------------------------------------------------------------------
# test_should_warm_never_called_api
# ---------------------------------------------------------------------------


def test_should_warm_never_called_api(group_timeline: GroupTimeline) -> None:
    # last_api_call_time is 0 by default — never called API
    for i in range(20):
        group_timeline.add("g1", role="user", content=f"msg{i}", speaker="A(1)")

    assert group_timeline.should_warm("g1", interval=10, ttl=3600) is False


# ---------------------------------------------------------------------------
# test_reset_warm_counter
# ---------------------------------------------------------------------------


def test_reset_warm_counter(group_timeline: GroupTimeline) -> None:
    group_timeline.set_input_tokens("g1", 1000)
    for i in range(10):
        group_timeline.add("g1", role="user", content=f"msg{i}", speaker="A(1)")

    assert group_timeline.should_warm("g1", interval=10, ttl=3600) is True

    group_timeline.reset_warm_counter("g1")
    assert group_timeline.should_warm("g1", interval=10, ttl=3600) is False


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
