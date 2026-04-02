"""Disk-based image cache: download, resize, store, load, cleanup.

Storage layout uses two-level hash directories (first 2 chars of file_id)
to prevent single-directory I/O degradation:

    storage/image_cache/ab/abc123def456.jpg
"""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from src.memory.types import ImageRefBlock

_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=15)
_MAX_CONCURRENT_DOWNLOADS = 8


class ImageCache:
    def __init__(self, cache_dir: Path | str, max_dimension: int = 768) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_dim = max_dimension
        self._sem = asyncio.Semaphore(_MAX_CONCURRENT_DOWNLOADS)

    def _path_for(self, file_id: str) -> Path:
        """Return the expected disk path for a given file_id."""
        bucket = file_id[:2]
        return self._dir / bucket / f"{file_id}.jpg"

    async def save(
        self,
        session: aiohttp.ClientSession,
        url: str,
        file_id: str,
    ) -> ImageRefBlock | None:
        """Download image, resize, cache to disk. Returns None on failure.

        If file_id already exists on disk, returns existing ref (cache hit).
        """
        if len(file_id) < 2:
            logger.warning("image file_id too short | file_id={!r}", file_id)
            return None

        path = self._path_for(file_id)

        # Cache hit — file already downloaded
        if path.exists():
            logger.debug("image cache hit | file_id={}", file_id)
            return ImageRefBlock(type="image_ref", path=str(path), media_type="image/jpeg")

        async with self._sem:
            t0 = time.perf_counter()
            try:
                async with session.get(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
                    if resp.status != 200:
                        logger.warning("image download failed | url={} status={}", url, resp.status)
                        return None
                    data = await resp.read()
            except Exception:
                logger.warning("image download error | url={}", url, exc_info=True)
                return None
            dl_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            try:
                ref = await asyncio.to_thread(self._process_and_save, data, path)
            except Exception:
                logger.warning("image processing error | file_id={}", file_id, exc_info=True)
                return None
            proc_ms = (time.perf_counter() - t1) * 1000

            logger.debug(
                "image save | file_id={} size={}KB download={:.0f}ms process={:.0f}ms",
                file_id, len(data) // 1024, dl_ms, proc_ms,
            )
            return ref

    def _process_and_save(self, data: bytes, path: Path) -> ImageRefBlock:
        """Resize image and save to disk as JPEG."""
        import pyvips

        img: Any = pyvips.Image.new_from_buffer(data, "")

        # For animated images (GIF), extract first frame
        if img.get_typeof("n-pages") != 0:
            n_pages = img.get("n-pages")
            if n_pages > 1:
                page_height = img.height // n_pages
                img = img.crop(0, 0, img.width, page_height)

        # Resize if needed
        max_side = max(img.width, img.height)
        if max_side > self._max_dim:
            scale = self._max_dim / max_side
            img = img.resize(scale)

        # Save
        path.parent.mkdir(parents=True, exist_ok=True)
        img.jpegsave(str(path), Q=80, strip=True)

        return ImageRefBlock(type="image_ref", path=str(path), media_type="image/jpeg")

    async def load_as_base64(self, ref: ImageRefBlock | dict[str, Any]) -> dict[str, Any] | None:
        """Read image from disk and return an Anthropic image content block.

        Returns None if the file no longer exists (expired/cleaned up).
        """
        path = Path(ref["path"])
        if not path.exists():
            return None

        t0 = time.perf_counter()
        data, b64 = await asyncio.to_thread(self._read_and_encode, path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "image load_base64 | file={} size={}KB elapsed={:.0f}ms",
            path.name, len(data) // 1024, elapsed_ms,
        )
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": ref["media_type"],
                "data": b64,
            },
        }

    @staticmethod
    def _read_and_encode(path: Path) -> tuple[bytes, str]:
        data = path.read_bytes()
        return data, base64.b64encode(data).decode("ascii")

    async def cleanup(self, max_age: timedelta = timedelta(hours=24)) -> None:
        """Delete cached images older than max_age. Remove empty subdirectories."""
        t0 = time.perf_counter()
        removed = await asyncio.to_thread(self._cleanup_sync, max_age)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if removed:
            logger.info("image_cache cleanup | removed={} elapsed={:.0f}ms", removed, elapsed_ms)

    def _cleanup_sync(self, max_age: timedelta) -> int:
        cutoff = time.time() - max_age.total_seconds()
        removed = 0

        for subdir in self._dir.iterdir():
            if not subdir.is_dir():
                continue
            for f in subdir.iterdir():
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            # Remove empty bucket directory
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()

        return removed
