import pytest

from src.identity.models import Identity
from src.llm.prompt import PromptBuilder, load_instruction
from src.memory.memo_store import MemoStore


@pytest.fixture
async def store(tmp_path) -> MemoStore:
    s = MemoStore(base_dir=str(tmp_path))
    await s.startup()
    await s.write("user_100", "测试用户｜test", "test")
    await s.write("group_200", "测试群｜test", "test")
    return s


@pytest.fixture
def identity() -> Identity:
    return Identity(id="test", name="Bot", personality="I am a bot.", proactive="Proactive rules.")


def test_load_instruction_missing(tmp_path) -> None:
    assert load_instruction(str(tmp_path)) == ""


def test_load_instruction_exists(tmp_path) -> None:
    (tmp_path / "instruction.md").write_text("Do things.")
    assert load_instruction(str(tmp_path)) == "Do things."


def test_build_static_called_once(identity: Identity) -> None:
    pb = PromptBuilder(instruction="Test instruction.")
    pb.build_static(identity, bot_self_id="999")
    assert pb.static_block is not None
    assert "I am a bot." in pb.static_block["text"]
    assert "Test instruction." in pb.static_block["text"]
    assert "Proactive rules." in pb.static_block["text"]
    assert pb.static_block["cache_control"] == {"type": "ephemeral"}


async def test_build_blocks_private(identity: Identity, store: MemoStore) -> None:
    pb = PromptBuilder(instruction="")
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="100", group_id=None, memo_store=store)
    assert len(blocks) == 2
    assert blocks[0] is pb.static_block
    assert "全局索引" in blocks[1]["text"]
    assert "私聊 @100" in blocks[1]["text"]
    assert "测试用户" in blocks[1]["text"]


async def test_build_blocks_group(identity: Identity, store: MemoStore) -> None:
    pb = PromptBuilder(instruction="")
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="100", group_id="200", memo_store=store)
    assert len(blocks) == 2
    assert "群 #200" in blocks[1]["text"]
    assert "测试群" in blocks[1]["text"]


async def test_static_block_shared_across_calls(identity: Identity, store: MemoStore) -> None:
    pb = PromptBuilder(instruction="")
    pb.build_static(identity, bot_self_id="999")
    b1 = await pb.build_blocks(user_id="100", group_id=None, memo_store=store)
    b2 = await pb.build_blocks(user_id="100", group_id="200", memo_store=store)
    assert b1[0] is b2[0]  # Same object reference = guaranteed cache hit


async def test_build_blocks_missing_memo(identity: Identity, store: MemoStore) -> None:
    """Non-existent entity should not crash, just show empty memo."""
    pb = PromptBuilder(instruction="")
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="999", group_id=None, memo_store=store)
    assert len(blocks) == 2
    assert "私聊 @999" in blocks[1]["text"]
