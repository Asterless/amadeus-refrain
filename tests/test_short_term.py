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


def test_max_rounds_eviction() -> None:
    mem = ShortTermMemory(max_rounds=2)  # 2 轮 = 4 条消息
    for i in range(5):
        mem.add("s1", "user", f"u{i}")
        mem.add("s1", "assistant", f"a{i}")
    msgs = mem.get("s1")
    assert len(msgs) == 4
    assert msgs[0]["content"] == "u3"


def test_clear(short_term: ShortTermMemory) -> None:
    short_term.add("s1", "user", "hello")
    short_term.clear("s1")
    assert short_term.get("s1") == []


def test_get_empty(short_term: ShortTermMemory) -> None:
    assert short_term.get("nonexistent") == []
