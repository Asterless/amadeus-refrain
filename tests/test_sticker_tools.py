"""Tests for SaveStickerTool and SendStickerTool."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.sticker.store import StickerStore
from src.tools.context import ToolContext
from src.tools.sticker_tools import SaveStickerTool, SendStickerTool

# ---------------------------------------------------------------------------
# Minimal valid image byte sequences (reused from test_sticker_store.py)
# ---------------------------------------------------------------------------

_JPEG_DATA = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"jpeg-payload-a"
_PNG_DATA = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64 + b"png-payload-a"
_GIF_DATA = b"GIF89a" + b"\x00" * 64 + b"gif-payload"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> StickerStore:
    return StickerStore(storage_dir=str(tmp_path / "stickers"))


@pytest.fixture
def superusers() -> set[str]:
    return {"admin1", "admin2"}


@pytest.fixture
def jpeg_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.jpg"
    p.write_bytes(_JPEG_DATA)
    return p


@pytest.fixture
def gif_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.gif"
    p.write_bytes(_GIF_DATA)
    return p


@pytest.fixture
def mock_bot() -> MagicMock:
    bot = MagicMock()
    bot.send_group_msg = AsyncMock()
    bot.send_private_msg = AsyncMock()
    return bot


# ---------------------------------------------------------------------------
# SaveStickerTool — schema
# ---------------------------------------------------------------------------


def test_save_sticker_schema(store: StickerStore, superusers: set[str]) -> None:
    tool = SaveStickerTool(store, superusers)
    schema = tool.parameters
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "image_ref" in props
    assert "description" in props
    assert "usage_hint" in props
    assert set(schema["required"]) == {"image_ref", "description", "usage_hint"}


def test_save_sticker_name_and_description(store: StickerStore, superusers: set[str]) -> None:
    tool = SaveStickerTool(store, superusers)
    assert tool.name == "save_sticker"
    assert "表情包" in tool.description


# ---------------------------------------------------------------------------
# SaveStickerTool — success
# ---------------------------------------------------------------------------


async def test_save_sticker_success(
    store: StickerStore, superusers: set[str], jpeg_file: Path
) -> None:
    tool = SaveStickerTool(store, superusers)
    ctx = ToolContext(user_id="regular_user")

    result = await tool.execute(ctx, image_ref=str(jpeg_file), description="开心笑", usage_hint="开心时发")

    assert "已收录" in result
    assert result.startswith("stk_")


async def test_save_sticker_returns_sticker_id(
    store: StickerStore, superusers: set[str], jpeg_file: Path
) -> None:
    tool = SaveStickerTool(store, superusers)
    ctx = ToolContext(user_id="regular_user")

    result = await tool.execute(ctx, image_ref=str(jpeg_file), description="开心笑", usage_hint="开心时发")

    # Extract sticker_id and verify it's in the store
    stk_id = result.split(" ")[0]
    assert store.get(stk_id) is not None


# ---------------------------------------------------------------------------
# SaveStickerTool — dedup
# ---------------------------------------------------------------------------


async def test_save_sticker_dedup(
    store: StickerStore, superusers: set[str], jpeg_file: Path
) -> None:
    tool = SaveStickerTool(store, superusers)
    ctx = ToolContext(user_id="regular_user")

    first = await tool.execute(ctx, image_ref=str(jpeg_file), description="第一次", usage_hint="hint")
    second = await tool.execute(ctx, image_ref=str(jpeg_file), description="第二次", usage_hint="hint")

    assert "已收录" in first
    assert "已存在" in second
    # The sticker_id should appear in both messages
    stk_id = first.split(" ")[0]
    assert stk_id in second


# ---------------------------------------------------------------------------
# SaveStickerTool — GIF rejected
# ---------------------------------------------------------------------------


async def test_save_sticker_gif_rejected(
    store: StickerStore, superusers: set[str], gif_file: Path
) -> None:
    tool = SaveStickerTool(store, superusers)
    ctx = ToolContext(user_id="regular_user")

    result = await tool.execute(ctx, image_ref=str(gif_file), description="gif", usage_hint="hint")

    assert "无法收录" in result
    assert "GIF" in result


# ---------------------------------------------------------------------------
# SaveStickerTool — missing file
# ---------------------------------------------------------------------------


async def test_save_sticker_missing_file(
    store: StickerStore, superusers: set[str]
) -> None:
    tool = SaveStickerTool(store, superusers)
    ctx = ToolContext(user_id="regular_user")

    result = await tool.execute(
        ctx,
        image_ref="/nonexistent/path/image.jpg",
        description="desc",
        usage_hint="hint",
    )

    assert "不存在" in result


# ---------------------------------------------------------------------------
# SaveStickerTool — admin source detection
# ---------------------------------------------------------------------------


async def test_save_sticker_admin_source(
    store: StickerStore, superusers: set[str], jpeg_file: Path
) -> None:
    tool = SaveStickerTool(store, superusers)
    ctx = ToolContext(user_id="admin1")

    result = await tool.execute(ctx, image_ref=str(jpeg_file), description="admin添加", usage_hint="hint")

    assert "已收录" in result
    stk_id = result.split(" ")[0]
    entry = store.get(stk_id)
    assert entry is not None
    assert entry["source"] == "admin"


async def test_save_sticker_non_admin_source(
    store: StickerStore, superusers: set[str], jpeg_file: Path
) -> None:
    tool = SaveStickerTool(store, superusers)
    ctx = ToolContext(user_id="regular_user")

    result = await tool.execute(ctx, image_ref=str(jpeg_file), description="auto添加", usage_hint="hint")

    assert "已收录" in result
    stk_id = result.split(" ")[0]
    entry = store.get(stk_id)
    assert entry is not None
    assert entry["source"] == "auto"


# ---------------------------------------------------------------------------
# SendStickerTool — schema
# ---------------------------------------------------------------------------


def test_send_sticker_schema(store: StickerStore) -> None:
    tool = SendStickerTool(store)
    schema = tool.parameters
    assert schema["type"] == "object"
    assert "sticker_id" in schema["properties"]
    assert schema["required"] == ["sticker_id"]


def test_send_sticker_name_and_description(store: StickerStore) -> None:
    tool = SendStickerTool(store)
    assert tool.name == "send_sticker"
    assert "表情包" in tool.description


# ---------------------------------------------------------------------------
# SendStickerTool — success (group)
# ---------------------------------------------------------------------------


async def test_send_sticker_group_success(
    store: StickerStore, mock_bot: MagicMock
) -> None:
    stk_id, _ = store.add(_JPEG_DATA, "desc", "hint")
    tool = SendStickerTool(store)
    ctx = ToolContext(bot=mock_bot, user_id="123456", group_id="987654")

    fake_seg = MagicMock()
    with patch("nonebot.adapters.onebot.v11.MessageSegment.image", return_value=fake_seg) as mock_image:
        result = await tool.execute(ctx, sticker_id=stk_id)

    assert f"已发送表情包 {stk_id}" == result
    mock_bot.send_group_msg.assert_awaited_once()
    mock_bot.send_private_msg.assert_not_awaited()

    # Verify the file path was passed to MessageSegment.image
    call_args = mock_image.call_args
    assert call_args is not None


# ---------------------------------------------------------------------------
# SendStickerTool — success (private)
# ---------------------------------------------------------------------------


async def test_send_sticker_private_success(
    store: StickerStore, mock_bot: MagicMock
) -> None:
    stk_id, _ = store.add(_JPEG_DATA, "desc", "hint")
    tool = SendStickerTool(store)
    ctx = ToolContext(bot=mock_bot, user_id="123456", group_id=None)

    fake_seg = MagicMock()
    with patch("nonebot.adapters.onebot.v11.MessageSegment.image", return_value=fake_seg):
        result = await tool.execute(ctx, sticker_id=stk_id)

    assert f"已发送表情包 {stk_id}" == result
    mock_bot.send_private_msg.assert_awaited_once()
    mock_bot.send_group_msg.assert_not_awaited()


# ---------------------------------------------------------------------------
# SendStickerTool — record_send is called
# ---------------------------------------------------------------------------


async def test_send_sticker_records_send(
    store: StickerStore, mock_bot: MagicMock
) -> None:
    stk_id, _ = store.add(_JPEG_DATA, "desc", "hint")
    assert store.get(stk_id)["send_count"] == 0  # type: ignore[index]

    tool = SendStickerTool(store)
    ctx = ToolContext(bot=mock_bot, user_id="123456", group_id=None)

    with patch("nonebot.adapters.onebot.v11.MessageSegment.image", return_value=MagicMock()):
        await tool.execute(ctx, sticker_id=stk_id)

    assert store.get(stk_id)["send_count"] == 1  # type: ignore[index]


# ---------------------------------------------------------------------------
# SendStickerTool — not found
# ---------------------------------------------------------------------------


async def test_send_sticker_not_found(
    store: StickerStore, mock_bot: MagicMock
) -> None:
    tool = SendStickerTool(store)
    ctx = ToolContext(bot=mock_bot, user_id="123456", group_id=None)

    result = await tool.execute(ctx, sticker_id="stk_00000000")

    assert "不存在" in result
    assert "stk_00000000" in result
    mock_bot.send_private_msg.assert_not_awaited()
    mock_bot.send_group_msg.assert_not_awaited()


# ---------------------------------------------------------------------------
# SendStickerTool — no bot
# ---------------------------------------------------------------------------


async def test_send_sticker_no_bot(store: StickerStore) -> None:
    stk_id, _ = store.add(_JPEG_DATA, "desc", "hint")
    tool = SendStickerTool(store)
    ctx = ToolContext(bot=None, user_id="123456", group_id=None)

    result = await tool.execute(ctx, sticker_id=stk_id)

    assert result == "Bot 不可用"


# ---------------------------------------------------------------------------
# SendStickerTool — send exception is caught
# ---------------------------------------------------------------------------


async def test_send_sticker_exception_handled(
    store: StickerStore, mock_bot: MagicMock
) -> None:
    stk_id, _ = store.add(_JPEG_DATA, "desc", "hint")
    mock_bot.send_group_msg.side_effect = RuntimeError("network error")

    tool = SendStickerTool(store)
    ctx = ToolContext(bot=mock_bot, user_id="123456", group_id="987654")

    with patch("nonebot.adapters.onebot.v11.MessageSegment.image", return_value=MagicMock()):
        result = await tool.execute(ctx, sticker_id=stk_id)

    assert "发送失败" in result
    assert stk_id in result
    # record_send should NOT have been called
    assert store.get(stk_id)["send_count"] == 0  # type: ignore[index]
