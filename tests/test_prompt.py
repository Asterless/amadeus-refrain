import pytest

from src.identity.models import Identity
from src.llm.prompt import PromptBuilder, load_instruction
from src.memory.memo_store import MemoStore
from src.sticker.store import StickerStore

# Minimal JPEG bytes (magic header + padding)
_JPEG_1PX = b"\xff\xd8\xff\xe0" + b"\x00" * 20


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


async def test_build_blocks_includes_sticker_index(
    identity: Identity, store: MemoStore, tmp_path: pytest.FixtureRequest
) -> None:
    """Sticker index appears in entity block when store has stickers."""
    sticker_store = StickerStore(storage_dir=str(tmp_path))
    sid, _ = sticker_store.add(_JPEG_1PX, "开心大笑", "搞笑时用")
    pb = PromptBuilder(instruction="", sticker_store=sticker_store)
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="100", group_id=None, memo_store=store)
    entity_text = blocks[1]["text"]
    assert "«表情包:stk_" in entity_text
    assert "开心大笑" in entity_text
    assert sid in entity_text


async def test_build_blocks_empty_sticker_store(
    identity: Identity, store: MemoStore, tmp_path: pytest.FixtureRequest
) -> None:
    """Empty sticker store injects the 'empty library' message."""
    sticker_store = StickerStore(storage_dir=str(tmp_path))
    pb = PromptBuilder(instruction="", sticker_store=sticker_store)
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="100", group_id=None, memo_store=store)
    entity_text = blocks[1]["text"]
    assert "当前表情包库为空" in entity_text


async def test_build_blocks_no_sticker_store(
    identity: Identity, store: MemoStore
) -> None:
    """No sticker store (default None) means no sticker text in entity block."""
    pb = PromptBuilder(instruction="")
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="100", group_id=None, memo_store=store)
    entity_text = blocks[1]["text"]
    assert "表情包" not in entity_text
