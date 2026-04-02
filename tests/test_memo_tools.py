"""Tests for RecallMemoTool and UpdateMemoTool."""

import asyncio

import pytest

from src.memory.memo_store import MemoStore
from src.tools.context import ToolContext
from src.tools.memo_tools import RecallMemoTool, UpdateMemoTool


@pytest.fixture
async def store_with_data(tmp_path) -> MemoStore:
    s = MemoStore(base_dir=str(tmp_path))
    await s.startup()
    await s.write("user_123456", "小明（明哥）｜杭州·后端\n\n喜欢 Go。", "test")
    await s.write("user_789012", "小红｜前端·Rust\n\n和 @123456 互怼。", "test")
    await s.write("group_987654", "技术群｜10人\n\n@123456 @789012 活跃。", "test")
    return s


async def test_recall_by_id(store_with_data: MemoStore) -> None:
    tool = RecallMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456")
    result = await tool.execute(ctx, id="user_123456")
    assert "小明" in result
    assert "Go" in result


async def test_recall_by_query(store_with_data: MemoStore) -> None:
    tool = RecallMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456")
    result = await tool.execute(ctx, query="小红")
    assert "789012" in result


async def test_recall_by_query_kind_filter(store_with_data: MemoStore) -> None:
    tool = RecallMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456")
    result = await tool.execute(ctx, query="活跃", kind="group")
    assert "987654" in result
    assert "user_" not in result


async def test_recall_not_found(store_with_data: MemoStore) -> None:
    tool = RecallMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456")
    result = await tool.execute(ctx, id="user_999999")
    assert "未找到" in result


async def test_recall_no_params(store_with_data: MemoStore) -> None:
    tool = RecallMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456")
    result = await tool.execute(ctx)
    assert "请提供" in result


async def test_update_memo_returns_immediately(store_with_data: MemoStore) -> None:
    tool = UpdateMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456")
    result = await tool.execute(ctx, id="user_123456", memo="小明｜新内容")
    assert "已提交" in result
    # Give the background task a moment to complete
    await asyncio.sleep(0.1)
    # Verify the write actually happened
    memo = store_with_data.read("user_123456")
    assert memo is not None
    assert "新内容" in memo.body


async def test_update_memo_error_handled_gracefully(store_with_data: MemoStore) -> None:
    """Invalid ID triggers error callback without crashing or losing the task reference."""
    tool = UpdateMemoTool(store_with_data)
    ctx = ToolContext(user_id="123456")
    result = await tool.execute(ctx, id="user_../../etc/passwd", memo="hacker")
    assert "已提交" in result
    await asyncio.sleep(0.1)
    # Task set should be empty (callback cleaned up)
    assert len(tool._background_tasks) == 0


async def test_update_memo_tool_schema() -> None:
    s = MemoStore(base_dir="/tmp/unused")
    tool = UpdateMemoTool(s)
    schema = tool.parameters
    assert "id" in schema["properties"]
    assert "memo" in schema["properties"]
    assert schema["required"] == ["id", "memo"]


async def test_recall_tool_schema() -> None:
    s = MemoStore(base_dir="/tmp/unused")
    tool = RecallMemoTool(s)
    schema = tool.parameters
    assert "id" in schema["properties"]
    assert "query" in schema["properties"]
    assert "kind" in schema["properties"]
