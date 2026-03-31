import pytest

from src.memory.long_term import LongTermMemory


@pytest.fixture
def mem(tmp_path: object) -> LongTermMemory:
    return LongTermMemory(memory_dir=str(tmp_path))


async def test_empty_profile(mem: LongTermMemory) -> None:
    assert await mem.get_profile("999") is None


async def test_update_and_get_profile(mem: LongTermMemory) -> None:
    await mem.update_profile("123", nickname="小明", traits={"爱好": "编程"})
    profile = await mem.get_profile("123")
    assert profile is not None
    assert profile["nickname"] == "小明"
    assert profile["traits"]["爱好"] == "编程"


async def test_profile_merge(mem: LongTermMemory) -> None:
    await mem.update_profile("123", nickname="小明", traits={"爱好": "编程"})
    await mem.update_profile("123", traits={"性格": "开朗"})
    profile = await mem.get_profile("123")
    assert profile is not None
    assert profile["nickname"] == "小明"
    assert profile["traits"]["爱好"] == "编程"
    assert profile["traits"]["性格"] == "开朗"


async def test_add_and_get_events(mem: LongTermMemory) -> None:
    await mem.add_event("123", "喜欢Python")
    await mem.add_event("123", "在学Rust")
    events = await mem.get_events("123", limit=10)
    assert len(events) == 2
    contents = {e["content"] for e in events}
    assert contents == {"喜欢Python", "在学Rust"}


async def test_events_limit(mem: LongTermMemory) -> None:
    for i in range(5):
        await mem.add_event("123", f"事件{i}")
    events = await mem.get_events("123", limit=2)
    assert len(events) == 2


async def test_full_context(mem: LongTermMemory) -> None:
    await mem.update_profile("123", nickname="小明")
    await mem.add_event("123", "测试事件")
    ctx = await mem.get_full_context("123")
    assert "小明" in ctx
    assert "测试事件" in ctx


async def test_qmd_parse_roundtrip(mem: LongTermMemory) -> None:
    """写入后重新解析，确保格式完整。"""
    await mem.update_profile("456", nickname="测试", traits={"a": "1", "b": "2"})
    await mem.add_event("456", "事件A")
    await mem.add_event("456", "事件B")

    # 重新读取
    profile = await mem.get_profile("456")
    events = await mem.get_events("456")
    assert profile is not None
    assert profile["nickname"] == "测试"
    assert len(profile["traits"]) == 2
    assert len(events) == 2


async def test_path_traversal_sanitized(mem: LongTermMemory) -> None:
    """user_id 含路径遍历字符应被过滤。"""
    await mem.update_profile("../../etc/passwd", nickname="hacker")
    profile = await mem.get_profile("../../etc/passwd")
    assert profile is not None
    assert profile["nickname"] == "hacker"
    # 文件名应该被清理
    path = mem._path("../../etc/passwd")
    assert ".." not in str(path.name)
