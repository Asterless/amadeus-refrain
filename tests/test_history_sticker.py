"""Tests for sticker recognition during history reload."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.history_loader import _extract_content
from src.memory.image_cache import ImageCache
from src.memory.types import ImageRefBlock
from src.sticker.store import StickerStore

# Minimal JPEG bytes (valid magic header + padding)
_JPEG_DATA = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"sticker-history-test"


def _make_mock_session(response_data: bytes, status: int = 200) -> MagicMock:
    """Build a mock aiohttp session supporting `async with session.get(url) as resp:`."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.read = AsyncMock(return_value=response_data)

    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_cm)
    return mock_session


def _is_image_ref_block(block: object) -> bool:
    """TypedDict cannot be used with isinstance; check by dict key."""
    return isinstance(block, dict) and block.get("type") == "image_ref"


@pytest.fixture
def sticker_store(tmp_path: Path) -> StickerStore:
    store = StickerStore(storage_dir=str(tmp_path / "stickers"))
    store.add(_JPEG_DATA, "测试表情", "开心时使用", source="auto")
    return store


@pytest.fixture
def image_cache(tmp_path: Path) -> ImageCache:
    return ImageCache(cache_dir=tmp_path / "image_cache")


async def test_history_sticker_recognition_uses_sticker_path(
    sticker_store: StickerStore,
    image_cache: ImageCache,
    tmp_path: Path,
) -> None:
    """Downloaded image matching a known sticker should resolve to the sticker path."""
    stk_id = sticker_store.lookup_by_hash(_JPEG_DATA)
    assert stk_id is not None
    sticker_path = sticker_store.resolve_path(stk_id)
    assert sticker_path is not None

    segments = [
        {"type": "image", "data": {"url": "http://example.com/img.jpg", "file": "abcdef1234567890"}},
    ]

    # Patch image_cache._process_and_save to write our JPEG bytes without pyvips
    original_process = image_cache._process_and_save

    def fake_process(data: bytes, path: Path) -> ImageRefBlock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return ImageRefBlock(type="image_ref", path=str(path), media_type="image/jpeg")

    image_cache._process_and_save = fake_process  # type: ignore[method-assign]

    mock_session = _make_mock_session(_JPEG_DATA)

    content = await _extract_content(segments, mock_session, image_cache, sticker_store)  # type: ignore[arg-type]

    # Restore original method
    image_cache._process_and_save = original_process  # type: ignore[method-assign]

    # Content should be a list of blocks (image only, no text)
    assert isinstance(content, list)
    assert len(content) == 1
    block = content[0]
    assert _is_image_ref_block(block)
    # Path should point to the sticker file, not the image_cache
    assert block["path"] == str(sticker_path)  # type: ignore[index]

    # The cached image file should have been removed (dedup cleanup)
    file_id = "abcdef1234567890"
    cached_path = image_cache._path_for(file_id)
    assert not cached_path.exists(), "Duplicate cache file should be removed after sticker match"


async def test_history_no_sticker_store_keeps_image(
    image_cache: ImageCache,
    tmp_path: Path,
) -> None:
    """Without a sticker_store, images are kept in image_cache as normal."""
    segments = [
        {"type": "image", "data": {"url": "http://example.com/img.jpg", "file": "abcdef1234567890"}},
    ]

    def fake_process(data: bytes, path: Path) -> ImageRefBlock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return ImageRefBlock(type="image_ref", path=str(path), media_type="image/jpeg")

    image_cache._process_and_save = fake_process  # type: ignore[method-assign]

    mock_session = _make_mock_session(_JPEG_DATA)

    content = await _extract_content(segments, mock_session, image_cache, sticker_store=None)  # type: ignore[arg-type]

    assert isinstance(content, list)
    assert len(content) == 1
    block = content[0]
    assert _is_image_ref_block(block)
    # Path should be inside the image_cache directory
    assert "image_cache" in block["path"]  # type: ignore[index]


async def test_history_unknown_image_not_in_sticker_store_kept(
    sticker_store: StickerStore,
    image_cache: ImageCache,
) -> None:
    """Image that doesn't match any sticker is kept in image_cache unchanged."""
    other_jpeg = b"\xff\xd8\xff\xe0" + b"\x01" * 64 + b"different-image"
    segments = [
        {"type": "image", "data": {"url": "http://example.com/img2.jpg", "file": "1234abcdef567890"}},
    ]

    def fake_process(data: bytes, path: Path) -> ImageRefBlock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return ImageRefBlock(type="image_ref", path=str(path), media_type="image/jpeg")

    image_cache._process_and_save = fake_process  # type: ignore[method-assign]

    mock_session = _make_mock_session(other_jpeg)

    content = await _extract_content(segments, mock_session, image_cache, sticker_store)  # type: ignore[arg-type]

    assert isinstance(content, list)
    assert len(content) == 1
    block = content[0]
    assert _is_image_ref_block(block)
    # Path should still be in image_cache (not the sticker store)
    assert "image_cache" in block["path"]  # type: ignore[index]
