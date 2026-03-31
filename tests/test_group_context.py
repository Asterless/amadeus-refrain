from src.memory.group_context import GroupContext


def test_add_and_format(group_context: GroupContext) -> None:
    group_context.add("g1", "100", "小明", "大家好")
    group_context.add("g1", "200", "小红", "你好呀")
    result = group_context.format("g1")
    assert "小明(100): 大家好" in result
    assert "小红(200): 你好呀" in result


def test_format_empty(group_context: GroupContext) -> None:
    assert group_context.format("g_nonexist") == ""


def test_max_messages() -> None:
    ctx = GroupContext(max_messages=3)
    for i in range(5):
        ctx.add("g1", str(i), f"user{i}", f"msg{i}")
    msgs = ctx.get_recent("g1")
    assert len(msgs) == 3
    assert msgs[0]["content"] == "msg2"


def test_group_isolation(group_context: GroupContext) -> None:
    group_context.add("g1", "100", "A", "群1消息")
    group_context.add("g2", "200", "B", "群2消息")
    r1 = group_context.format("g1")
    r2 = group_context.format("g2")
    assert "群1消息" in r1
    assert "群2消息" not in r1
    assert "群2消息" in r2


def test_get_recent_limit(group_context: GroupContext) -> None:
    for i in range(8):
        group_context.add("g1", "100", "A", f"msg{i}")
    msgs = group_context.get_recent("g1", limit=3)
    assert len(msgs) == 3
    assert msgs[0]["content"] == "msg5"
