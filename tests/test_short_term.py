from src.memory.short_term import ShortTermMemory


def test_add_and_get(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "你好")
    short_term.add("s1", "assistant", "你好呀")
    msgs = short_term.get("s1")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "你好呀"


def test_session_isolation(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "消息1")
    short_term.add("s2", "user", "消息2")
    assert len(short_term.get("s1")) == 1
    assert len(short_term.get("s2")) == 1
    assert short_term.get("s1")[0]["content"] == "消息1"


def test_clear(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "hello")
    short_term.clear("s1")
    assert short_term.get("s1") == []


def test_get_empty(short_term: ShortTermMemory) -> None:
    assert short_term.get("nonexistent") == []


def test_summary_empty_by_default(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "hello")
    assert short_term.get_summary("s1") == ""
    assert short_term.get_summary("nonexistent") == ""


def test_input_tokens_tracking(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "hello")
    assert short_term.get_input_tokens("s1") == 0
    short_term.set_input_tokens("s1", 5000)
    assert short_term.get_input_tokens("s1") == 5000
    # 不存在的 session 不报错
    assert short_term.get_input_tokens("nonexistent") == 0


def test_needs_compact(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "hello")
    short_term.set_input_tokens("s1", 150_000)
    assert short_term.needs_compact("s1", max_tokens=200_000, ratio=0.7)  # 150k > 140k
    assert not short_term.needs_compact("s1", max_tokens=200_000, ratio=0.8)  # 150k < 160k
    assert not short_term.needs_compact("nonexistent", max_tokens=200_000, ratio=0.7)


def test_compact(short_term: ShortTermMemory) -> None:
    for i in range(10):
        short_term.add("s1", "user", f"u{i}")
        short_term.add("s1", "assistant", f"a{i}")
    short_term.set_input_tokens("s1", 99999)

    # compact 前半（10条消息）
    short_term.compact("s1", split=10, new_summary="对话摘要：用户打了10次招呼")
    msgs = short_term.get("s1")
    assert len(msgs) == 10
    assert msgs[0]["content"] == "u5"
    assert short_term.get_summary("s1") == "对话摘要：用户打了10次招呼"
    assert short_term.get_input_tokens("s1") == 0  # compact 后重置


def test_compact_preserves_accumulation(short_term: ShortTermMemory) -> None:
    """compact 后可以继续累积消息。"""
    short_term.add("s1", "user", "u0")
    short_term.add("s1", "assistant", "a0")
    short_term.compact("s1", split=1, new_summary="摘要")

    short_term.add("s1", "user", "u1")
    msgs = short_term.get("s1")
    assert len(msgs) == 2  # a0 + u1
    assert msgs[0]["content"] == "a0"
    assert short_term.get_summary("s1") == "摘要"


def test_messages_accumulate_without_limit(short_term: ShortTermMemory) -> None:
    """消息不再有硬上限，可以一直累积。"""
    for i in range(100):
        short_term.add("s1", "user", f"msg{i}")
    msgs = short_term.get("s1")
    assert len(msgs) == 100
    assert msgs[0]["content"] == "msg0"
    assert msgs[99]["content"] == "msg99"


def test_max_sessions_eviction() -> None:
    """超过 500 个 session 时，移除最旧的。"""
    mem = ShortTermMemory()
    for i in range(501):
        mem.add(f"s{i}", "user", f"msg{i}")
    # s0 被移除
    assert mem.get("s0") == []
    assert len(mem.get("s1")) == 1
    assert len(mem.get("s500")) == 1
