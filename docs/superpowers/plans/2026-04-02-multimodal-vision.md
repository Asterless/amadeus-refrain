# Multimodal Vision Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the bot to understand images and QQ stickers sent by users via Claude Vision, with disk-based image caching and graceful degradation.

**Architecture:** Messages are parsed into content blocks (text + image refs). Images are downloaded and cached to disk keyed by QQ file ID. At API call time, image refs are loaded from disk, base64-encoded, and sent as Anthropic image content blocks. QQ face emojis are mapped to Chinese text labels.

**Tech Stack:** pyvips (image resize), aiohttp (download), OneBot V11 segments, Anthropic Vision API

**Spec:** `docs/superpowers/specs/2026-04-02-multimodal-vision-design.md`

---

### Task 1: Add VisionConfig to BotConfig

**Files:**
- Modify: `src/config.py:86-99`
- Modify: `config.example.toml`
- Modify: `tests/test_config_loader.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_config_loader.py`, add a test for default vision config:

```python
def test_vision_config_defaults() -> None:
    from src.config import VisionConfig

    v = VisionConfig()
    assert v.enabled is True
    assert v.max_images_per_message == 5
    assert v.max_images_per_request == 15
    assert v.max_dimension == 768
    assert v.cache_dir == "storage/image_cache"
    assert v.cache_max_age_hours == 24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_loader.py::test_vision_config_defaults -v`
Expected: FAIL — `ImportError: cannot import name 'VisionConfig'`

- [ ] **Step 3: Add VisionConfig model and wire into BotConfig**

In `src/config.py`, add `VisionConfig` before `BotConfig`:

```python
class VisionConfig(BaseModel):
    """多模态视觉配置。"""

    enabled: bool = True
    max_images_per_message: int = 5
    max_images_per_request: int = 15
    max_dimension: int = 768
    cache_dir: str = "storage/image_cache"
    cache_max_age_hours: int = 24
```

Add `vision: VisionConfig = VisionConfig()` to `BotConfig`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_loader.py::test_vision_config_defaults -v`
Expected: PASS

- [ ] **Step 5: Write test for VisionConfig loaded from TOML**

In `tests/test_config_loader.py`, add:

```python
def test_vision_config_from_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOT_CONFIG_PATH", raising=False)
    toml_file = tmp_path / "config.toml"
    _write_toml(
        toml_file,
        """
[vision]
enabled = false
max_images_per_message = 3
max_images_per_request = 10
max_dimension = 512
cache_dir = "custom/cache"
cache_max_age_hours = 12
""",
    )
    cfg = load_config(config_path=str(toml_file))
    assert cfg.vision.enabled is False
    assert cfg.vision.max_images_per_message == 3
    assert cfg.vision.max_images_per_request == 10
    assert cfg.vision.max_dimension == 512
    assert cfg.vision.cache_dir == "custom/cache"
    assert cfg.vision.cache_max_age_hours == 12
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_config_loader.py::test_vision_config_from_toml -v`
Expected: PASS (Pydantic handles TOML loading automatically)

- [ ] **Step 7: Update config.example.toml**

Add at the end of `config.example.toml`:

```toml
# ---------------------------------------------------------------------------
# 多模态视觉（图片理解）
# ---------------------------------------------------------------------------
[vision]
# 总开关，关闭后所有图片降级为 [图片] 文字
enabled = true

# 单条消息最多处理的图片数，超出的用 [图片] 代替
max_images_per_message = 5

# 单次 API 请求累计图片上限（跨所有历史消息）
max_images_per_request = 15

# 图片缩放最大边像素
max_dimension = 768

# 图片缓存目录
cache_dir = "storage/image_cache"

# 缓存过期时间（小时）
cache_max_age_hours = 24
```

- [ ] **Step 8: Run full config tests and lint**

Run: `uv run pytest tests/test_config_loader.py -v && uv run ruff check src/config.py`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/config.py config.example.toml tests/test_config_loader.py
git commit -m "feat: add VisionConfig for multimodal image settings"
```

---

### Task 2: QQ Face Emoji Mapping

**Files:**
- Create: `src/constants/__init__.py`
- Create: `src/constants/qq_face.py`
- Create: `tests/test_qq_face.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qq_face.py`:

```python
"""QQ Face emoji mapping tests."""

from src.constants.qq_face import QQ_FACE, face_to_text


def test_known_face_ids() -> None:
    assert QQ_FACE[0] == "惊讶"
    assert QQ_FACE[14] == "微笑"
    assert QQ_FACE[178] == "捂脸"


def test_face_to_text_known() -> None:
    assert face_to_text(14) == "[微笑]"
    assert face_to_text(178) == "[捂脸]"


def test_face_to_text_unknown() -> None:
    assert face_to_text(99999) == "[表情]"


def test_mapping_has_common_faces() -> None:
    """Ensure the mapping covers the most commonly used QQ faces."""
    common_ids = [0, 1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 21, 32, 49, 53, 78, 79]
    for fid in common_ids:
        assert fid in QQ_FACE, f"Missing common face id {fid}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qq_face.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.constants'`

- [ ] **Step 3: Create the module**

Create `src/constants/__init__.py` (empty file).

Create `src/constants/qq_face.py`:

```python
"""QQ classic face emoji ID → Chinese name mapping.

Reference: QQ face IDs from OneBot V11 / NapCat protocol.
Face segments arrive as {"type": "face", "data": {"id": "14"}}.
Unknown IDs fall back to generic [表情].
"""

QQ_FACE: dict[int, str] = {
    0: "惊讶",
    1: "撇嘴",
    2: "色",
    3: "发呆",
    4: "得意",
    5: "流泪",
    6: "害羞",
    7: "闭嘴",
    8: "睡",
    9: "大哭",
    10: "尴尬",
    11: "发怒",
    12: "调皮",
    13: "呲牙",
    14: "微笑",
    15: "难过",
    16: "酷",
    18: "抓狂",
    19: "吐",
    20: "偷笑",
    21: "可爱",
    22: "白眼",
    23: "傲慢",
    24: "饥饿",
    25: "困",
    26: "惊恐",
    27: "流汗",
    28: "憨笑",
    29: "悠闲",
    30: "奋斗",
    31: "咒骂",
    32: "疑问",
    33: "嘘",
    34: "晕",
    36: "衰",
    37: "骷髅",
    38: "敲打",
    39: "再见",
    42: "鼓掌",
    46: "右哼哼",
    49: "委屈",
    53: "吓",
    54: "可怜",
    55: "菜刀",
    56: "啤酒",
    59: "咖啡",
    60: "饭",
    61: "猪头",
    62: "玫瑰",
    63: "凋谢",
    66: "心碎",
    67: "蛋糕",
    69: "炸弹",
    74: "月亮",
    75: "太阳",
    76: "彩虹",
    77: "拥抱",
    78: "强",
    79: "弱",
    85: "差劲",
    86: "爱你",
    89: "爱情",
    96: "回头",
    97: "跳绳",
    98: "挥手",
    99: "激动",
    104: "双喜",
    105: "鞭炮",
    106: "灯笼",
    107: "发财",
    108: "K歌",
    109: "购物",
    111: "帅",
    116: "喝奶",
    118: "香蕉",
    119: "飞机",
    120: "开车",
    123: "高铁右车头",
    124: "多云",
    125: "下雨",
    126: "钞票",
    127: "熊猫",
    128: "灯泡",
    130: "闹钟",
    131: "打伞",
    133: "钻戒",
    136: "药",
    137: "手枪",
    138: "青蛙",
    144: "喝彩",
    146: "笑哭",
    147: "我最美",
    171: "点赞",
    172: "托脸",
    173: "拜托",
    174: "无奈",
    175: "不看",
    176: "惊喜",
    177: "生日快乐",
    178: "捂脸",
    179: "奸笑",
    180: "嗨",
    181: "打call",
    182: "变形",
    183: "仔细分析",
    212: "托腮",
    214: "666",
    219: "发怒",
    222: "汪汪",
    225: "心碎",
    226: "菜狗",
    227: "崇拜",
    228: "比心",
    229: "庆祝",
    230: "老色批",
    231: "吃糖",
    232: "惊吓",
    233: "生气",
    240: "敬礼",
    241: "狂笑",
    242: "面无表情",
    243: "摸鱼",
    244: "魔鬼笑",
    245: "哦",
    246: "请",
    247: "睁眼",
    260: "搬砖",
    261: "忙到飞起",
    262: "脑阔疼",
    263: "沧桑",
    264: "捂脸哭",
    265: "辣眼睛",
    266: "哦哟",
    267: "头秃",
    268: "问号脸",
    269: "暗中观察",
    270: "emm",
    271: "吃瓜",
    272: "呵呵哒",
    273: "我酸了",
    274: "汪汪",
    276: "无语",
    277: "敬礼",
    278: "面无表情",
    281: "摸锦鲤",
    282: "期待",
    284: "拿到红包",
    285: "真好",
    287: "拜谢",
    289: "元宝",
    290: "牛啊",
    293: "打工人",
    294: "右亲亲",
    297: "拒绝",
    298: "啵啵",
    299: "嫌弃",
    305: "右拜年",
    306: "牛气冲天",
    307: "喵喵",
    308: "求红包",
    312: "NO",
    313: "拽",
    314: "我看看",
    315: "流泪",
    318: "666",
    319: "裂开",
    320: "暴富",
    322: "翻白眼",
    324: "让我看看",
    326: "OK",
    332: "举牌牌",
    334: "社会社会",
    336: "打招呼",
    341: "调皮",
}


def face_to_text(face_id: int) -> str:
    """Convert a QQ face ID to a bracketed Chinese text label."""
    name = QQ_FACE.get(face_id, "表情")
    return f"[{name}]"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_qq_face.py -v`
Expected: All PASS

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/constants/ && uv run pyright src/constants/`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add src/constants/ tests/test_qq_face.py
git commit -m "feat: add QQ face emoji ID to text mapping"
```

---

### Task 3: Content Block Types

**Files:**
- Create: `src/memory/types.py`

- [ ] **Step 1: Create the types module**

Create `src/memory/types.py`:

```python
"""Shared content block types for multimodal message storage.

Messages with images use a list[ContentBlock] instead of a plain str.
ImageRefBlock stores a disk path (not base64) to keep memory low.
Conversion to Anthropic API format happens at request assembly time.
"""

from typing import Literal, TypedDict


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ImageRefBlock(TypedDict):
    type: Literal["image_ref"]
    path: str  # disk path, e.g. "storage/image_cache/ab/ab3f7c.jpg"
    media_type: str  # e.g. "image/jpeg"


ContentBlock = TextBlock | ImageRefBlock

# Messages store content as str (text-only, backward compat) or list of blocks (multimodal).
Content = str | list[ContentBlock]
```

- [ ] **Step 2: Lint and type-check**

Run: `uv run ruff check src/memory/types.py && uv run pyright src/memory/types.py`
Expected: Clean

- [ ] **Step 3: Commit**

```bash
git add src/memory/types.py
git commit -m "feat: add content block types for multimodal messages"
```

---

### Task 4: Image Cache Module

**Files:**
- Create: `src/memory/image_cache.py`
- Create: `tests/test_image_cache.py`
- Modify: `pyproject.toml` (add pyvips dependency)

- [ ] **Step 1: Add pyvips dependency**

In `pyproject.toml`, add `"pyvips>=2.2.0"` to the `dependencies` list:

```toml
dependencies = [
    "nonebot2[fastapi]>=2.4.0",
    "nonebot-adapter-onebot>=2.4.0",
    "pydantic>=2.0.0",
    "httpx>=0.27.0",
    "loguru>=0.7.0",
    "aiofiles>=25.1.0",
    "aiohttp>=3.13.4",
    "tenacity>=9.1.4",
    "pyvips>=2.2.0",
]
```

Run: `uv sync`

Note: This requires `libvips` system library. On Debian/Ubuntu: `sudo apt-get install libvips-dev`. If not installed, pyvips import will fail — that's OK, tests can still run with mocking. The Dockerfile will be updated in Task 11.

- [ ] **Step 2: Write failing tests**

Create `tests/test_image_cache.py`:

```python
"""Image cache module tests."""

import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.memory.image_cache import ImageCache


@pytest.fixture
def cache(tmp_path: Path) -> ImageCache:
    return ImageCache(cache_dir=tmp_path, max_dimension=256)


def _write_test_image(path: Path) -> None:
    """Write a minimal valid JPEG to disk for testing."""
    # Smallest valid JPEG: SOI + APP0 + minimal scan + EOI
    # For tests that just need a file to exist, raw bytes suffice.
    # For resize tests, use pyvips to create a real image.
    import pyvips

    img = pyvips.Image.black(100, 80).copy(interpretation="srgb")
    img.jpegsave(str(path), Q=50)


class TestSaveAndLoad:
    async def test_save_downloads_and_caches(self, cache: ImageCache, tmp_path: Path) -> None:
        """save() should download image, resize, and store to disk."""
        import pyvips

        # Create a test image as bytes
        img = pyvips.Image.black(1024, 768).copy(interpretation="srgb")
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
        # Pre-create the cached file
        subdir = tmp_path / "ab"
        subdir.mkdir()
        cached_file = subdir / "abc123.jpg"
        _write_test_image(cached_file)

        mock_session = AsyncMock()
        ref = await cache.save(mock_session, url="http://example.com/img.jpg", file_id="abc123")

        assert ref is not None
        # Should NOT have called session.get — cache hit
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
        assert len(block["source"]["data"]) > 0  # base64 string

    def test_load_as_base64_missing_file(self, cache: ImageCache) -> None:
        ref = {"type": "image_ref", "path": "/nonexistent/file.jpg", "media_type": "image/jpeg"}
        block = cache.load_as_base64(ref)
        assert block is None

    async def test_resize_respects_max_dimension(self, cache: ImageCache, tmp_path: Path) -> None:
        """Images larger than max_dimension should be scaled down."""
        import pyvips

        img = pyvips.Image.black(2000, 1000).copy(interpretation="srgb")
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

        saved = pyvips.Image.new_from_file(ref["path"])
        assert max(saved.width, saved.height) <= 256


class TestCleanup:
    def test_cleanup_removes_old_files(self, cache: ImageCache, tmp_path: Path) -> None:
        subdir = tmp_path / "ab"
        subdir.mkdir()

        old_file = subdir / "old.jpg"
        new_file = subdir / "new.jpg"
        _write_test_image(old_file)
        _write_test_image(new_file)

        # Make old_file appear old by setting mtime
        old_mtime = time.time() - 3600 * 25  # 25 hours ago
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_image_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.memory.image_cache'`

- [ ] **Step 4: Implement ImageCache**

Create `src/memory/image_cache.py`:

```python
"""Disk-based image cache: download, resize, store, load, cleanup.

Storage layout uses two-level hash directories (first 2 chars of file_id)
to prevent single-directory I/O degradation:

    storage/image_cache/ab/abc123def456.jpg
"""

from __future__ import annotations

import base64
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from src.memory.types import ImageRefBlock

_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=15)


class ImageCache:
    def __init__(self, cache_dir: Path | str, max_dimension: int = 768) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_dim = max_dimension

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
        path = self._path_for(file_id)

        # Cache hit — file already downloaded
        if path.exists():
            return ImageRefBlock(type="image_ref", path=str(path), media_type="image/jpeg")

        try:
            async with session.get(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
                if resp.status != 200:
                    logger.warning("image download failed | url={} status={}", url, resp.status)
                    return None
                data = await resp.read()
        except Exception:
            logger.warning("image download error | url={}", url, exc_info=True)
            return None

        try:
            return self._process_and_save(data, path)
        except Exception:
            logger.warning("image processing error | file_id={}", file_id, exc_info=True)
            return None

    def _process_and_save(self, data: bytes, path: Path) -> ImageRefBlock:
        """Resize image and save to disk as JPEG."""
        import pyvips

        img = pyvips.Image.new_from_buffer(data, "")

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

    def load_as_base64(self, ref: ImageRefBlock | dict[str, Any]) -> dict[str, Any] | None:
        """Read image from disk and return an Anthropic image content block.

        Returns None if the file no longer exists (expired/cleaned up).
        """
        path = Path(ref["path"])
        if not path.exists():
            return None

        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": ref["media_type"],
                "data": b64,
            },
        }

    def cleanup(self, max_age: timedelta = timedelta(hours=24)) -> None:
        """Delete cached images older than max_age. Remove empty subdirectories."""
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

        if removed:
            logger.info("image_cache cleanup | removed={}", removed)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_image_cache.py -v`
Expected: All PASS

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src/memory/image_cache.py && uv run pyright src/memory/image_cache.py`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/memory/image_cache.py tests/test_image_cache.py
git commit -m "feat: add ImageCache for disk-based image caching with pyvips resize"
```

---

### Task 5: Update ShortTermMemory for Content Blocks

**Files:**
- Modify: `src/memory/short_term.py:1-34`
- Modify: `tests/test_short_term.py`

- [ ] **Step 1: Write failing test for multimodal content**

In `tests/test_short_term.py`, add:

```python
from src.memory.types import ContentBlock, ImageRefBlock, TextBlock


def test_add_content_blocks(short_term: ShortTermMemory) -> None:
    """Content can be a list of content blocks (multimodal)."""
    blocks: list[ContentBlock] = [
        TextBlock(type="text", text="look at this"),
        ImageRefBlock(type="image_ref", path="storage/image_cache/ab/abc.jpg", media_type="image/jpeg"),
    ]
    short_term.add("s1", "user", blocks)
    msgs = short_term.get("s1")
    assert len(msgs) == 1
    assert isinstance(msgs[0]["content"], list)
    assert msgs[0]["content"][0]["type"] == "text"
    assert msgs[0]["content"][1]["type"] == "image_ref"


def test_mixed_str_and_blocks(short_term: ShortTermMemory) -> None:
    """Str and block content can coexist in the same session."""
    short_term.add("s1", "user", "plain text")
    blocks: list[ContentBlock] = [TextBlock(type="text", text="with image")]
    short_term.add("s1", "assistant", blocks)
    msgs = short_term.get("s1")
    assert isinstance(msgs[0]["content"], str)
    assert isinstance(msgs[1]["content"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_short_term.py::test_add_content_blocks -v`
Expected: FAIL — type error or wrong signature

- [ ] **Step 3: Update ShortTermMemory**

In `src/memory/short_term.py`:

Change the import and `ChatMessage` type:

```python
"""短期记忆：每个会话累积对话历史，按需 compact。"""

from typing import Literal, TypedDict

from src.memory.types import Content

_MAX_SESSIONS = 500


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: Content
```

Change `add()` signature:

```python
def add(self, session_id: str, role: Literal["user", "assistant"], content: Content) -> None:
    state = self._get_or_create(session_id)
    state.messages.append(ChatMessage(role=role, content=content))
```

No other methods need changes — they just pass `content` through.

- [ ] **Step 4: Run all short_term tests**

Run: `uv run pytest tests/test_short_term.py -v`
Expected: All PASS (existing tests pass strings, new tests pass blocks)

- [ ] **Step 5: Commit**

```bash
git add src/memory/short_term.py tests/test_short_term.py
git commit -m "feat: ShortTermMemory accepts multimodal content blocks"
```

---

### Task 6: Update GroupTimeline for Content Blocks

**Files:**
- Modify: `src/memory/group_timeline.py:1-101`
- Modify: `tests/test_group_timeline.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_group_timeline.py`, add:

```python
from src.memory.types import ContentBlock, ImageRefBlock, TextBlock


def test_add_content_blocks(group_timeline: GroupTimeline) -> None:
    blocks: list[ContentBlock] = [
        TextBlock(type="text", text="看这个"),
        ImageRefBlock(type="image_ref", path="storage/image_cache/ab/abc.jpg", media_type="image/jpeg"),
    ]
    group_timeline.add("g1", role="user", content=blocks, speaker="Alice(123)")
    msgs = group_timeline.get_messages("g1")
    assert len(msgs) == 1
    assert isinstance(msgs[0]["content"], list)


def test_to_anthropic_merges_multimodal_users(group_timeline: GroupTimeline) -> None:
    """Consecutive user messages with content blocks should merge correctly."""
    blocks1: list[ContentBlock] = [
        TextBlock(type="text", text="看图"),
        ImageRefBlock(type="image_ref", path="cache/ab/abc.jpg", media_type="image/jpeg"),
    ]
    group_timeline.add("g1", role="user", content=blocks1, speaker="Alice(1)")
    group_timeline.add("g1", role="user", content="纯文本", speaker="Bob(2)")

    msgs = group_timeline.to_anthropic_messages("g1")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    # Merged content should be a list of blocks
    content = msgs[0]["content"]
    assert isinstance(content, list)
    # First user's text block gets speaker prefix
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Alice(1): 看图"
    # Image block follows
    assert content[1]["type"] == "image_ref"
    # Second user's text
    assert content[2]["type"] == "text"
    assert content[2]["text"] == "Bob(2): 纯文本"


def test_to_anthropic_str_and_blocks_mixed(group_timeline: GroupTimeline) -> None:
    """A mix of str and block content in the timeline should all merge."""
    group_timeline.add("g1", role="user", content="hello", speaker="A(1)")
    blocks: list[ContentBlock] = [
        TextBlock(type="text", text="看"),
        ImageRefBlock(type="image_ref", path="cache/img.jpg", media_type="image/jpeg"),
    ]
    group_timeline.add("g1", role="user", content=blocks, speaker="B(2)")
    group_timeline.add("g1", role="assistant", content="OK")

    msgs = group_timeline.to_anthropic_messages("g1")
    assert len(msgs) == 2  # merged users + assistant
    # First message is merged users — must be a list since one has images
    assert isinstance(msgs[0]["content"], list)
    assert msgs[1]["content"] == "OK"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_group_timeline.py::test_add_content_blocks -v`
Expected: FAIL — type error

- [ ] **Step 3: Update GroupTimeline**

In `src/memory/group_timeline.py`:

Update imports and `TimelineMessage`:

```python
"""群聊统一时间线：合并 GroupContext 与群组的 ShortTermMemory。"""

from typing import Any, Literal, TypedDict

from src.memory.types import Content, ContentBlock, TextBlock

_MAX_GROUPS = 200


class TimelineMessage(TypedDict):
    role: Literal["user", "assistant"]
    speaker: str | None  # user → "昵称(QQ号)", assistant → None
    content: Content
```

Update `add()`:

```python
def add(
    self,
    group_id: str,
    *,
    role: Literal["user", "assistant"],
    content: Content,
    speaker: str | None = None,
) -> None:
    """追加一条消息；超出上限时淘汰最旧的消息。"""
    state = self._get_or_create(group_id)
    state.messages.append(TimelineMessage(role=role, speaker=speaker, content=content))
    if len(state.messages) > state._max:
        state.messages = state.messages[-state._max :]
```

Rewrite `to_anthropic_messages()` to handle content blocks:

```python
def to_anthropic_messages(self, group_id: str) -> list[dict[str, Any]]:
    """将时间线转为 Anthropic messages 格式。

    连续的 user 消息合并为一个 block。
    当所有消息都是纯文本时，合并为 str（保持现有行为）。
    当任何消息含图片时，合并为 list[ContentBlock]。
    """
    messages = self.get_messages(group_id)
    if not messages:
        return []

    result: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "assistant":
            result.append({"role": "assistant", "content": msg["content"]})
            i += 1
        else:
            # Collect consecutive user messages
            user_batch: list[TimelineMessage] = []
            while i < len(messages) and messages[i]["role"] == "user":
                user_batch.append(messages[i])
                i += 1
            result.append({"role": "user", "content": _merge_user_contents(user_batch)})

    return result
```

Add the helper function `_merge_user_contents` at module level (above the class):

```python
def _merge_user_contents(batch: list[TimelineMessage]) -> Content:
    """Merge consecutive user messages into a single content value.

    Returns str if all messages are plain text (backward compatible).
    Returns list[ContentBlock] if any message contains image blocks.
    """
    has_blocks = any(isinstance(m["content"], list) for m in batch)

    if not has_blocks:
        # All plain text — merge into a single string
        lines: list[str] = []
        for m in batch:
            assert isinstance(m["content"], str)
            if m["speaker"] is not None:
                lines.append(f"{m['speaker']}: {m['content']}")
            else:
                lines.append(m["content"])
        return "\n".join(lines)

    # At least one has blocks — merge into a flat list of ContentBlock
    merged: list[ContentBlock] = []
    for m in batch:
        prefix = f"{m['speaker']}: " if m["speaker"] is not None else ""
        if isinstance(m["content"], str):
            merged.append(TextBlock(type="text", text=f"{prefix}{m['content']}"))
        else:
            # Prepend speaker prefix to the first text block
            for j, block in enumerate(m["content"]):
                if j == 0 and block["type"] == "text" and prefix:
                    merged.append(TextBlock(type="text", text=f"{prefix}{block['text']}"))
                else:
                    merged.append(block)
    return merged
```

- [ ] **Step 4: Run all group_timeline tests**

Run: `uv run pytest tests/test_group_timeline.py -v`
Expected: All PASS (existing + new)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/memory/group_timeline.py && uv run pyright src/memory/group_timeline.py`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add src/memory/group_timeline.py tests/test_group_timeline.py
git commit -m "feat: GroupTimeline supports multimodal content blocks with merge"
```

---

### Task 7: Update Message Parsing — `_render_message()`

**Files:**
- Modify: `src/plugins/chat/__init__.py:164-186`

This step changes `_render_message()` to extract face emojis and produce content blocks when images are present. Image download is async, so the function becomes a coroutine.

Note: Testing `_render_message()` directly is hard because it depends on NoneBot `Message` objects. We'll test the face mapping via `test_qq_face.py` and the image flow via integration in Task 10. The actual wiring test will be done in Task 10.

- [ ] **Step 1: Update imports in plugin**

In `src/plugins/chat/__init__.py`, add these imports at the top:

```python
from src.constants.qq_face import face_to_text
from src.memory.image_cache import ImageCache
from src.memory.types import Content, ContentBlock, ImageRefBlock, TextBlock
```

- [ ] **Step 2: Add ImageCache and VisionConfig to plugin init**

Add global variables:

```python
_image_cache: ImageCache
_vision_enabled: bool = True
_max_images_per_message: int = 5
```

In `_init()`, after loading `bot_config`, add:

```python
from src.config import VisionConfig

_image_cache = ImageCache(
    cache_dir=bot_config.vision.cache_dir,
    max_dimension=bot_config.vision.max_dimension,
)
_vision_enabled = bot_config.vision.enabled
_max_images_per_message = bot_config.vision.max_images_per_message

# Cleanup stale cache on startup
_image_cache.cleanup(max_age=timedelta(hours=bot_config.vision.cache_max_age_hours))
```

Also add `from datetime import timedelta` to imports.

- [ ] **Step 3: Rewrite `_render_message()` to async and multimodal**

Replace the existing `_render_message()` with:

```python
async def _render_message(
    msg: Message,
    reply: object | None = None,
    session: aiohttp.ClientSession | None = None,
) -> Content:
    """将消息段转为文本或内容块列表。

    Returns plain str if no images; list[ContentBlock] if images present.
    """
    text_parts: list[str] = []
    images: list[ImageRefBlock] = []
    image_count = 0

    # 引用回复 → [回复 昵称(QQ号): 原文摘要]
    if reply is not None:
        sender = getattr(reply, "sender", None)
        reply_msg = getattr(reply, "message", None)
        if sender and reply_msg:
            uid = getattr(sender, "user_id", "") or ""
            nick = getattr(sender, "nickname", "") or str(uid)
            original = reply_msg.extract_plain_text().strip()
            if len(original) > _REPLY_PREVIEW_MAX:
                original = original[:_REPLY_PREVIEW_MAX] + "…"
            text_parts.append(f"[回复 {nick}({uid}): {original}] ")

    for seg in msg:
        if seg.type == "text":
            text_parts.append(seg.data.get("text", ""))
        elif seg.type == "at":
            qq = seg.data.get("qq", "")
            text_parts.append(f"@{qq}")
        elif seg.type == "face":
            face_id = seg.data.get("id", "")
            try:
                text_parts.append(face_to_text(int(face_id)))
            except (ValueError, TypeError):
                text_parts.append("[表情]")
        elif seg.type == "image" and _vision_enabled and session is not None:
            if image_count >= _max_images_per_message:
                text_parts.append("[图片]")
                continue
            url = seg.data.get("url", "")
            file_id = seg.data.get("file", "")
            if url and file_id:
                # Sanitize file_id: remove extension suffix like ".image"
                file_id = file_id.split(".")[0] if "." in file_id else file_id
                ref = await _image_cache.save(session, url=url, file_id=file_id)
                if ref is not None:
                    images.append(ref)
                    image_count += 1
                else:
                    text_parts.append("[图片]")
            else:
                text_parts.append("[图片]")
        elif seg.type == "image":
            text_parts.append("[图片]")

    text = "".join(text_parts).strip()

    if not images:
        return text

    # Build content blocks: text first, then images
    blocks: list[ContentBlock] = []
    if text:
        blocks.append(TextBlock(type="text", text=text))
    blocks.extend(images)
    return blocks
```

- [ ] **Step 4: Update callers**

In `collect_group_context()`, change to async render:

```python
@group_listener.handle()
async def collect_group_context(bot: Bot, event: GroupMessageEvent) -> None:
    if _allowed_groups and event.group_id not in _allowed_groups:
        return
    if str(event.user_id) == bot.self_id:
        return
    content = await _render_message(event.get_message(), reply=event.reply, session=_llm._session)
    if not content:
        return

    nickname = event.sender.nickname or str(event.user_id)
    group_id = str(event.group_id)
    _timeline.add(
        group_id,
        role="user",
        speaker=f"{nickname}({event.user_id})",
        content=content,
    )

    _scheduler.notify(group_id, is_at=event.is_tome())
```

In `handle_private_chat()`, change to async render and pass content to chat():

```python
@private_chat.handle()
async def handle_private_chat(bot: Bot, event: MessageEvent) -> None:
    if isinstance(event, GroupMessageEvent):
        return
    if _allowed_private_users and event.user_id not in _allowed_private_users:
        return

    reply_msg = getattr(event, "reply", None)
    user_content = await _render_message(event.get_message(), reply=reply_msg, session=_llm._session)
    if not user_content:
        return

    sid = _session_id(event)
    identity = _identity_mgr.resolve()
    ctx = ToolContext(bot=bot, user_id=str(event.user_id), group_id=None, session_id=sid)

    async def send_segment(text: str) -> None:
        await bot.send(event, Message(text))

    try:
        reply = await _llm.chat(
            session_id=sid,
            user_id=str(event.user_id),
            user_content=user_content,
            identity=identity,
            group_id=None,
            ctx=ctx,
            on_segment=send_segment,
        )
    except Exception:
        logger.exception("chat error")
        reply = "出错了，请稍后再试"

    if _dream_enabled:
        await _dream.maybe_run(_dream_llm_call)

    if reply:
        await private_chat.finish(Message(reply))
```

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/plugins/chat/__init__.py`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add src/plugins/chat/__init__.py
git commit -m "feat: _render_message extracts face emoji and image segments"
```

---

### Task 8: Update History Loader for Images

**Files:**
- Modify: `src/memory/history_loader.py`

- [ ] **Step 1: Update history loader to extract face and image segments**

Replace the content extraction loop in `_load_one_group()`:

```python
"""启动时从 NapCat HTTP API 拉取群历史消息，填充群聊上下文。"""

from __future__ import annotations

from typing import Any

import aiohttp
from loguru import logger

from src.constants.qq_face import face_to_text
from src.memory.group_timeline import GroupTimeline
from src.memory.image_cache import ImageCache
from src.memory.types import Content, ContentBlock, ImageRefBlock, TextBlock


async def load_group_history(
    napcat_url: str,
    group_ids: list[str],
    timeline: GroupTimeline,
    count: int = 30,
    bot_self_id: str = "",
    image_cache: ImageCache | None = None,
) -> None:
    """从 NapCat 拉取多个群的历史消息。"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        for gid in group_ids:
            try:
                await _load_one_group(session, napcat_url, gid, timeline, count, bot_self_id, image_cache)
            except Exception:
                logger.warning("load_history failed | group={}", gid, exc_info=True)


async def _load_one_group(
    session: aiohttp.ClientSession,
    napcat_url: str,
    group_id: str,
    timeline: GroupTimeline,
    count: int,
    bot_self_id: str = "",
    image_cache: ImageCache | None = None,
) -> None:
    async with session.post(
        f"{napcat_url}/get_group_msg_history",
        json={"group_id": int(group_id), "count": count},
    ) as resp:
        data: dict[str, Any] = await resp.json()

    if data.get("retcode") != 0:
        logger.warning("get_group_msg_history error | group={} resp={}", group_id, data.get("message", ""))
        return

    messages: list[dict[str, Any]] = data.get("data", {}).get("messages", [])
    if not messages:
        return

    loaded = 0

    for msg in messages:
        sender: dict[str, Any] = msg.get("sender", {})
        user_id = str(sender.get("user_id", ""))
        nickname = sender.get("nickname", "") or sender.get("card", "") or user_id

        content = await _extract_content(msg.get("message", []), session, image_cache)
        if not content:
            continue

        if bot_self_id and user_id == bot_self_id:
            timeline.add(group_id, role="assistant", content=content)
        else:
            timeline.add(group_id, role="user", speaker=f"{nickname}({user_id})", content=content)
        loaded += 1

    logger.info("history loaded | group={} messages={}", group_id, loaded)


async def _extract_content(
    segments: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    image_cache: ImageCache | None,
) -> Content:
    """Extract text, face, and image segments into a Content value."""
    text_parts: list[str] = []
    images: list[ImageRefBlock] = []

    for seg in segments:
        seg_type = seg.get("type", "")
        seg_data: dict[str, Any] = seg.get("data", {})

        if seg_type == "text":
            text_parts.append(seg_data.get("text", ""))
        elif seg_type == "face":
            face_id = seg_data.get("id", "")
            try:
                text_parts.append(face_to_text(int(face_id)))
            except (ValueError, TypeError):
                text_parts.append("[表情]")
        elif seg_type == "image" and image_cache is not None:
            url = seg_data.get("url", "")
            file_id = seg_data.get("file", "")
            if url and file_id:
                file_id = file_id.split(".")[0] if "." in file_id else file_id
                ref = await image_cache.save(session, url=url, file_id=file_id)
                if ref is not None:
                    images.append(ref)
                else:
                    text_parts.append("[图片]")
            else:
                text_parts.append("[图片]")
        elif seg_type == "image":
            text_parts.append("[图片]")

    text = "".join(text_parts).strip()

    if not images:
        return text

    blocks: list[ContentBlock] = []
    if text:
        blocks.append(TextBlock(type="text", text=text))
    blocks.extend(images)
    return blocks
```

- [ ] **Step 2: Update the call site in `__init__.py`**

In `src/plugins/chat/__init__.py`, update `_on_connect()` to pass `image_cache`:

```python
await load_group_history(
    napcat_url=bot_config.napcat.api_url,
    group_ids=group_ids,
    timeline=_timeline,
    count=bot_config.group.history_load_count,
    bot_self_id=bot.self_id,
    image_cache=_image_cache if _vision_enabled else None,
)
```

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/memory/history_loader.py src/plugins/chat/__init__.py`
Expected: Clean

- [ ] **Step 4: Commit**

```bash
git add src/memory/history_loader.py src/plugins/chat/__init__.py
git commit -m "feat: history loader extracts face emoji and image segments"
```

---

### Task 9: Update LLM Client for Content Blocks

**Files:**
- Modify: `src/llm/client.py:62-68, 283-335, 340-351`
- Modify: `tests/test_client.py`

- [ ] **Step 1: Write failing test for content block conversion**

In `tests/test_client.py`, add at the top-level:

```python
from src.memory.types import ContentBlock, ImageRefBlock, TextBlock
from src.llm.client import _to_anthropic_message
from src.memory.short_term import ChatMessage


def test_to_anthropic_message_str() -> None:
    """String content passes through unchanged."""
    msg = ChatMessage(role="user", content="hello")
    result = _to_anthropic_message(msg)
    assert result == {"role": "user", "content": "hello"}


def test_to_anthropic_message_blocks() -> None:
    """Block content passes through as-is (image_ref converted later)."""
    blocks: list[ContentBlock] = [
        TextBlock(type="text", text="look"),
        ImageRefBlock(type="image_ref", path="cache/ab/abc.jpg", media_type="image/jpeg"),
    ]
    msg = ChatMessage(role="user", content=blocks)
    result = _to_anthropic_message(msg)
    assert result["role"] == "user"
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_client.py::test_to_anthropic_message_blocks -v`
Expected: FAIL — content is a list but `_to_anthropic_message` expects str

- [ ] **Step 3: Update `_to_anthropic_message` and `chat()` signature**

In `src/llm/client.py`:

Update imports:

```python
from src.memory.image_cache import ImageCache
from src.memory.types import Content, ContentBlock, ImageRefBlock, TextBlock
```

Replace `_to_anthropic_message`:

```python
def _to_anthropic_message(msg: ChatMessage) -> dict[str, Any]:
    return {"role": msg["role"], "content": msg["content"]}
```

(The function now just passes through — Content can be str or list. The actual image_ref → base64 conversion happens in `_resolve_image_refs`.)

Add a new function to convert image refs to base64:

```python
def _resolve_image_refs(
    messages: list[dict[str, Any]],
    image_cache: ImageCache | None,
    max_images: int,
) -> list[dict[str, Any]]:
    """Convert image_ref blocks to Anthropic image blocks (base64).

    Enforces a per-request image cap. Oldest images are replaced first.
    """
    if image_cache is None:
        return messages

    # First pass: find all image_ref positions
    image_positions: list[tuple[int, int]] = []  # (msg_index, block_index)
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for bi, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "image_ref":
                    image_positions.append((mi, bi))

    # Determine which images to keep (newest first → keep last N)
    keep_set = set(image_positions[-max_images:]) if len(image_positions) > max_images else set(image_positions)

    # Second pass: resolve or replace
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[dict[str, Any]] = []
        for bi, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "image_ref":
                new_content.append(block)
                continue
            if (mi, bi) not in keep_set:
                new_content.append({"type": "text", "text": "[图片]"})
                continue
            resolved = image_cache.load_as_base64(block)
            if resolved is not None:
                # Preserve cache_control if the original block had it
                if "cache_control" in block:
                    resolved = {**resolved, "cache_control": block["cache_control"]}
                new_content.append(resolved)
            else:
                new_content.append({"type": "text", "text": "[图片已过期]"})
        msg["content"] = new_content

    return messages
```

- [ ] **Step 3b: Add `_content_text` helper for compact methods**

The compact methods (`_compact` and `_compact_group`) build conversation text from `msg["content"]`. With multimodal content, this would stringify the block list. Add a helper at module level:

```python
def _content_text(content: Content) -> str:
    """Extract plain text from Content, ignoring image blocks."""
    if isinstance(content, str):
        return content
    return " ".join(b["text"] for b in content if b.get("type") == "text")
```

Then update `_compact()` (line ~539):

```python
for msg in history[:split]:
    role_label = "用户" if msg["role"] == "user" else "助手"
    lines.append(f"{role_label}: {_content_text(msg['content'])}")
```

And `_compact_group()` (line ~635):

```python
for msg in messages[:split]:
    if msg["role"] == "assistant":
        lines.append(f"{identity.name}: {_content_text(msg['content'])}")
    elif msg["speaker"]:
        lines.append(f"{msg['speaker']}: {_content_text(msg['content'])}")
    else:
        lines.append(f"用户: {_content_text(msg['content'])}")
```

- [ ] **Step 4: Update `chat()` method**

Change the `chat()` signature — rename `user_text` to `user_content` with type `Content`:

```python
async def chat(
    self,
    session_id: str,
    user_id: str,
    user_content: Content,
    identity: Identity,
    group_id: str | None = None,
    ctx: ToolContext | None = None,
    on_segment: Callable[[str], Awaitable[None]] | None = None,
) -> str | None:
```

Update the logging line:

```python
content_preview = user_content[:80] if isinstance(user_content, str) else str(user_content)[:80]
logger.info("chat | session={} user={} identity={} text={!r}", session_id, user_id, identity.id, content_preview)
```

Update the private chat branch to use `user_content`:

```python
else:
    # Private: use ShortTermMemory
    self._short_term.add(session_id, "user", user_content)
```

Add image resolution before the API call loop. After building `messages` and `system_blocks`, add:

```python
messages = _resolve_image_refs(messages, self._image_cache, self._max_images_per_request)
```

- [ ] **Step 5: Add `image_cache` and `max_images_per_request` to `LLMClient.__init__`**

Add parameters to `__init__`:

```python
def __init__(
    self,
    ...
    image_cache: ImageCache | None = None,
    max_images_per_request: int = 15,
) -> None:
    ...
    self._image_cache = image_cache
    self._max_images_per_request = max_images_per_request
```

- [ ] **Step 6: Update `_build_private_messages` cache breakpoint for content blocks**

In `_build_private_messages`, the cache breakpoint wraps content in `[_cached_text(...)]`. This only works for str content. Update:

```python
def _build_private_messages(self, session_id: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    summary = self._short_term.get_summary(session_id)
    if summary:
        messages.append({
            "role": "user",
            "content": [_cached_text(f"[对话摘要]\n{summary}")],
        })
        messages.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

    history = self._short_term.get(session_id)
    for i, msg in enumerate(history):
        m = _to_anthropic_message(msg)
        if i == len(history) - 2:
            content = m["content"]
            if isinstance(content, str):
                m = {"role": m["role"], "content": [_cached_text(content)]}
            elif isinstance(content, list):
                # Add cache_control to last block
                content = [*content]
                content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
                m = {"role": m["role"], "content": content}
        messages.append(m)

    return messages
```

- [ ] **Step 7: Update `_build_group_messages` cache breakpoint similarly**

In `_build_group_messages`, update the cache breakpoint handling:

```python
# Place cache breakpoint at the position recorded by the previous API call
cached_idx = self._timeline.get_cached_msg_index(group_id)
if 0 < cached_idx < len(messages):
    target = messages[cached_idx]
    content = target.get("content")
    if isinstance(content, str):
        messages[cached_idx] = {"role": target["role"], "content": [_cached_text(content)]}
    elif isinstance(content, list):
        content = [*content]
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
        messages[cached_idx] = {"role": target["role"], "content": content}
```

- [ ] **Step 8: Run all client tests**

Run: `uv run pytest tests/test_client.py -v`
Expected: All PASS

- [ ] **Step 9: Lint and type-check**

Run: `uv run ruff check src/llm/client.py && uv run pyright src/llm/client.py`
Expected: Clean

- [ ] **Step 10: Commit**

```bash
git add src/llm/client.py tests/test_client.py
git commit -m "feat: LLM client resolves image_ref blocks to Anthropic vision format"
```

---

### Task 10: Wire ImageCache into Plugin and LLMClient

**Files:**
- Modify: `src/plugins/chat/__init__.py:39-106`

- [ ] **Step 1: Pass ImageCache to LLMClient**

In `_init()`, update the `LLMClient` constructor call to pass image_cache:

```python
_llm = LLMClient(
    base_url=bot_config.llm.base_url,
    api_key=bot_config.llm.api_key,
    model=bot_config.llm.model,
    prompt_builder=prompt_builder,
    short_term=short_term,
    tools=tools,
    max_context_tokens=bot_config.llm.context.max_context_tokens,
    micro_ratio=bot_config.compact.micro_ratio,
    full_ratio=bot_config.compact.full_ratio,
    max_compact_failures=bot_config.compact.max_failures,
    cache_hit_warn=bot_config.compact.cache_hit_warn,
    group_timeline=_timeline,
    memo_store=memo_store,
    on_compact=lambda: _dream.notify_compact(),
    image_cache=_image_cache if _vision_enabled else None,
    max_images_per_request=bot_config.vision.max_images_per_request,
)
```

- [ ] **Step 2: Update the `chat()` call in scheduler**

The `GroupChatScheduler` calls `_llm.chat()`. Check `src/llm/scheduler.py` for the `user_text` parameter and update to `user_content`. The scheduler passes `user_text=""` for group chats (the content is already in the timeline). Update the parameter name in the scheduler's call to `user_content=""`.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All PASS

- [ ] **Step 4: Lint everything**

Run: `uv run ruff check src/ && uv run pyright`
Expected: Clean

- [ ] **Step 5: Commit**

```bash
git add src/plugins/chat/__init__.py src/llm/scheduler.py
git commit -m "feat: wire ImageCache into plugin init and LLMClient"
```

---

### Task 11: Dockerfile and Dependencies

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Add libvips to Dockerfile**

In the `Dockerfile`, add `libvips` installation to both builder and runtime stages. Update the builder stage:

```dockerfile
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends libvips-dev && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libvips && rm -rf /var/lib/apt/lists/*

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

WORKDIR /app
COPY --from=builder /app /app

CMD [".venv/bin/python", "bot.py"]
```

Note: `libvips-dev` in builder (for compiling pyvips), `libvips` in runtime (just the shared library).

- [ ] **Step 2: Verify Docker build**

Run: `docker build -t qq-bot-test .`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: add libvips to Dockerfile for image processing"
```

---

### Task 12: Full Integration Verification

- [ ] **Step 1: Run complete test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 2: Run lint and type-check**

Run: `uv run ruff check src/ && uv run pyright`
Expected: Clean

- [ ] **Step 3: Verify no regressions in existing behavior**

Run: `uv run pytest tests/test_short_term.py tests/test_group_timeline.py tests/test_client.py tests/test_config_loader.py -v`
Expected: All existing tests still PASS — string content is backward compatible

- [ ] **Step 4: Final commit if any loose changes**

```bash
git status
# If any unstaged changes remain, commit them
```
