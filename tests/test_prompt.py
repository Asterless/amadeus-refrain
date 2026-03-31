"""Prompt 构建测试。"""

from src.identity.models import Identity
from src.llm.prompt import TOOL_GUIDE, PromptBuilder
from src.memory.group_context import GroupContext
from src.memory.long_term import LongTermMemory


async def test_build_basic(tmp_path: object) -> None:
    long_term = LongTermMemory(memory_dir=str(tmp_path))
    group_ctx = GroupContext()
    builder = PromptBuilder(long_term=long_term, group_context=group_ctx)

    identity = Identity(id="test", name="测试", personality="你是测试人设。")
    blocks = await builder.build_blocks(identity, user_id="123")

    assert len(blocks) >= 1
    assert "你是测试人设。" in blocks[0]["text"]
    assert TOOL_GUIDE.strip() in blocks[0]["text"]
    assert blocks[0].get("cache_control") == {"type": "ephemeral"}


async def test_build_with_memory(tmp_path: object) -> None:
    long_term = LongTermMemory(memory_dir=str(tmp_path))
    await long_term.update_profile("123", nickname="小明")

    group_ctx = GroupContext()
    builder = PromptBuilder(long_term=long_term, group_context=group_ctx)

    identity = Identity(id="test", name="测试", personality="人设。")
    blocks = await builder.build_blocks(identity, user_id="123")

    assert len(blocks) == 2
    assert "小明" in blocks[1]["text"]
    assert blocks[1].get("cache_control") == {"type": "ephemeral"}


async def test_build_with_group_id(tmp_path: object) -> None:
    """群号写入 system blocks，但群聊记录不在 system 里（由 client 注入 messages）。"""
    long_term = LongTermMemory(memory_dir=str(tmp_path))
    group_ctx = GroupContext()
    group_ctx.add("g1", "200", "小红", "今天天气好")

    builder = PromptBuilder(long_term=long_term, group_context=group_ctx)
    identity = Identity(id="test", name="测试", personality="人设。")
    blocks = await builder.build_blocks(identity, user_id="123", group_id="g1")

    # 群号在 system，但群聊记录不在
    assert len(blocks) == 1
    assert "群 g1" in blocks[0]["text"]
    assert "今天天气好" not in blocks[0]["text"]

    # 群聊记录通过 build_context_message 获取
    ctx_msg = builder.build_context_message("g1")
    assert ctx_msg is not None
    assert "今天天气好" in ctx_msg


async def test_build_no_group_context_for_private(tmp_path: object) -> None:
    long_term = LongTermMemory(memory_dir=str(tmp_path))
    group_ctx = GroupContext()
    group_ctx.add("g1", "200", "小红", "群消息")

    builder = PromptBuilder(long_term=long_term, group_context=group_ctx)
    identity = Identity(id="test", name="测试", personality="人设。")
    blocks = await builder.build_blocks(identity, user_id="123", group_id=None)

    assert len(blocks) == 1
    assert "群消息" not in blocks[0]["text"]
