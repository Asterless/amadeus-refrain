"""Image cache module tests."""

import time
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.memory.image_cache import ImageCache


@pytest.fixture
def cache(tmp_path: Path) -> ImageCache:
    return ImageCache(cache_dir=tmp_path, max_dimension=256)


def _write_test_image(path: Path) -> None:
    """Write a minimal valid JPEG to disk for testing."""
    import pyvips  # type: ignore[import-untyped]

    img: Any = pyvips.Image.black(100, 80).copy(interpretation="srgb")  # pyright: ignore[reportOptionalMemberAccess,reportCallIssue]
    img.jpegsave(str(path), Q=50)


class TestSaveAndLoad:
    async def test_save_downloads_and_caches(self, cache: ImageCache, tmp_path: Path) -> None:
        """save() should download image, resize, and store to disk."""
        import pyvips  # type: ignore[import-untyped]

        img: Any = pyvips.Image.black(1024, 768).copy(interpretation="srgb")  # pyright: ignore[reportOptionalMemberAccess,reportCallIssue]
        buf = img.jpegsave_buffer(Q=80)

        mock_resp = AsyncMock()
        mock_resp.read = AsyncMock(return_value=buf)
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_resp)

        ref = await cache.save(mock_session, url="http://example.com/img.jpg", file_id="abc123def456")

        assert ref is not None
        assert ref["path"].endswith(".jpg")
        assert ref["media_type"] == "image/jpeg"
        assert Path(ref["path"]).exists()

        # Verify two-level directory structure
        assert "/ab/" in ref["path"] or "\\ab\\" in ref["path"]

    async def test_save_returns_none_on_download_failure(self, cache: ImageCache) -> None:
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_resp)

        ref = await cache.save(mock_session, url="http://example.com/missing.jpg", file_id="deadbeef")
        assert ref is None

    async def test_save_skips_if_cached(self, cache: ImageCache, tmp_path: Path) -> None:
        """If file_id already exists on disk, return existing ref without downloading."""
        subdir = tmp_path / "ab"
        subdir.mkdir()
        cached_file = subdir / "abc123.jpg"
        _write_test_image(cached_file)

        mock_session = AsyncMock()
        ref = await cache.save(mock_session, url="http://example.com/img.jpg", file_id="abc123")

        assert ref is not None
        mock_session.get.assert_not_called()

    def test_load_as_base64(self, cache: ImageCache, tmp_path: Path) -> None:
        subdir = tmp_path / "ab"
        subdir.mkdir()
        img_path = subdir / "abc123.jpg"
        _write_test_image(img_path)

        ref = {"type": "image_ref", "path": str(img_path), "media_type": "image/jpeg"}
        block = cache.load_as_base64(ref)

        assert block is not None
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/jpeg"
        assert len(block["source"]["data"]) > 0

    def test_load_as_base64_missing_file(self, cache: ImageCache) -> None:
        ref = {"type": "image_ref", "path": "/nonexistent/file.jpg", "media_type": "image/jpeg"}
        block = cache.load_as_base64(ref)
        assert block is None

    async def test_resize_respects_max_dimension(self, cache: ImageCache, tmp_path: Path) -> None:
        """Images larger than max_dimension should be scaled down."""
        import pyvips  # type: ignore[import-untyped]

        img: Any = pyvips.Image.black(2000, 1000).copy(interpretation="srgb")  # pyright: ignore[reportOptionalMemberAccess,reportCallIssue]
        buf = img.jpegsave_buffer(Q=80)

        mock_resp = AsyncMock()
        mock_resp.read = AsyncMock(return_value=buf)
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_resp)

        ref = await cache.save(mock_session, url="http://example.com/big.jpg", file_id="bigimg001")
        assert ref is not None

        saved: Any = pyvips.Image.new_from_file(ref["path"])
        assert max(saved.width, saved.height) <= 256


class TestCleanup:
    def test_cleanup_removes_old_files(self, cache: ImageCache, tmp_path: Path) -> None:
        subdir = tmp_path / "ab"
        subdir.mkdir()

        old_file = subdir / "old.jpg"
        new_file = subdir / "new.jpg"
        _write_test_image(old_file)
        _write_test_image(new_file)

        old_mtime = time.time() - 3600 * 25
        import os

        os.utime(old_file, (old_mtime, old_mtime))

        cache.cleanup(max_age=timedelta(hours=24))

        assert not old_file.exists()
        assert new_file.exists()

    def test_cleanup_removes_empty_subdirs(self, cache: ImageCache, tmp_path: Path) -> None:
        subdir = tmp_path / "cd"
        subdir.mkdir()
        old_file = subdir / "only.jpg"
        _write_test_image(old_file)

        old_mtime = time.time() - 3600 * 25
        import os

        os.utime(old_file, (old_mtime, old_mtime))

        cache.cleanup(max_age=timedelta(hours=24))

        assert not old_file.exists()
        assert not subdir.exists()
