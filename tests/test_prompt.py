"""Prompt 构建测试。"""

from src.identity.models import Identity
from src.llm.prompt import PromptBuilder
from src.memory.long_term import LongTermMemory

_TEST_INSTRUCTION = "【测试指令】这是测试用指令。"


async def test_build_basic(tmp_path: object) -> None:
    long_term = LongTermMemory(memory_dir=str(tmp_path))
    builder = PromptBuilder(long_term=long_term, instruction=_TEST_INSTRUCTION)

    identity = Identity(id="test", name="测试", personality="你是测试人设。")
    blocks = await builder.build_blocks(identity, user_id="123")

    assert len(blocks) >= 1
    assert "你是测试人设。" in blocks[0]["text"]
    assert _TEST_INSTRUCTION in blocks[0]["text"]
    assert blocks[0].get("cache_control") == {"type": "ephemeral"}


async def test_build_without_instruction(tmp_path: object) -> None:
    long_term = LongTermMemory(memory_dir=str(tmp_path))
    builder = PromptBuilder(long_term=long_term)

    identity = Identity(id="test", name="测试", personality="你是测试人设。")
    blocks = await builder.build_blocks(identity, user_id="123")

    assert len(blocks) >= 1
    assert blocks[0]["text"] == "你是测试人设。"


async def test_build_with_memory(tmp_path: object) -> None:
    long_term = LongTermMemory(memory_dir=str(tmp_path))
    await long_term.update_profile("123", nickname="小明")

    builder = PromptBuilder(long_term=long_term, instruction=_TEST_INSTRUCTION)

    identity = Identity(id="test", name="测试", personality="人设。")
    blocks = await builder.build_blocks(identity, user_id="123")

    assert len(blocks) == 2
    assert "小明" in blocks[1]["text"]
    assert blocks[1].get("cache_control") == {"type": "ephemeral"}


async def test_build_with_group_id(tmp_path: object) -> None:
    """群号写入 system blocks，但群聊记录不在 system 里（由 client 注入 messages）。"""
    long_term = LongTermMemory(memory_dir=str(tmp_path))

    builder = PromptBuilder(long_term=long_term)
    identity = Identity(id="test", name="测试", personality="人设。")
    blocks = await builder.build_blocks(identity, user_id="123", group_id="g1")

    # 群号在 system
    assert len(blocks) == 1
    assert "群 g1" in blocks[0]["text"]


async def test_build_no_group_context_for_private(tmp_path: object) -> None:
    long_term = LongTermMemory(memory_dir=str(tmp_path))

    builder = PromptBuilder(long_term=long_term)
    identity = Identity(id="test", name="测试", personality="人设。")
    blocks = await builder.build_blocks(identity, user_id="123", group_id=None)

    assert len(blocks) == 1
    assert "当前在群" not in blocks[0]["text"]


async def test_proactive_injected_in_group(long_term: LongTermMemory) -> None:
    """identity.proactive should be appended to system blocks in group chat."""
    identity = Identity(
        id="test", name="测试", personality="性格描述",
        proactive="【主动发言规则】积极参与群聊",
    )
    builder = PromptBuilder(long_term=long_term, instruction="")
    blocks = await builder.build_blocks(identity=identity, user_id="u1", group_id="g1")
    assert "【主动发言规则】积极参与群聊" in blocks[0]["text"]


async def test_proactive_not_injected_without_group(long_term: LongTermMemory) -> None:
    """identity.proactive should NOT be injected for private chat."""
    identity = Identity(
        id="test", name="测试", personality="性格描述",
        proactive="【主动发言规则】积极参与群聊",
    )
    builder = PromptBuilder(long_term=long_term, instruction="")
    blocks = await builder.build_blocks(identity=identity, user_id="u1", group_id=None)
    assert "【主动发言规则】" not in blocks[0]["text"]


async def test_proactive_not_injected_when_none(long_term: LongTermMemory) -> None:
    """No proactive rules when identity.proactive is None."""
    identity = Identity(
        id="test", name="测试", personality="性格描述",
        proactive=None,
    )
    builder = PromptBuilder(long_term=long_term, instruction="")
    blocks = await builder.build_blocks(identity=identity, user_id="u1", group_id="g1")
    assert "主动发言" not in blocks[0]["text"]
