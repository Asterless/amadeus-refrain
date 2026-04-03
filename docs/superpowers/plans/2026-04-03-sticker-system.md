# Sticker System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the bot a self-evolving sticker library — it can collect, send, and autonomously curate stickers, with Dream handling periodic cleanup.

**Architecture:** New `StickerStore` class manages the flat file storage + JSON index in `storage/stickers/`. Two conversation tools (`save_sticker`, `send_sticker`) let the LLM collect and send stickers. `PromptBuilder` injects the sticker index into the entity block. `DreamAgent` gains two extra tools for cleanup. History loader recognizes known stickers to avoid duplicate storage.

**Tech Stack:** Python 3, Pydantic config, asyncio, pyvips (existing), aiohttp (existing), NoneBot2 OneBot V11 adapter.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/sticker/store.py` | `StickerStore`: load/save index, add/remove/lookup stickers, hash computation, GIF detection |
| Create | `src/sticker/__init__.py` | Re-export `StickerStore` |
| Create | `src/tools/sticker_tools.py` | `SaveStickerTool` and `SendStickerTool` |
| Create | `tests/test_sticker_store.py` | Tests for `StickerStore` |
| Create | `tests/test_sticker_tools.py` | Tests for sticker tools |
| Modify | `src/config.py` | Add `StickerConfig` and wire into `BotConfig` |
| Modify | `config.example.toml` | Add `[sticker]` section |
| Modify | `src/llm/prompt.py` | Inject sticker index into entity block |
| Modify | `tests/test_prompt.py` | Test sticker index injection |
| Modify | `src/plugins/chat/__init__.py` | Register sticker tools, pass `StickerStore` |
| Modify | `src/llm/dream.py` | Add `list_stickers` / `delete_sticker` tools + prompt extension |
| Modify | `tests/test_dream.py` | Test dream sticker tools |
| Modify | `src/memory/history_loader.py` | Recognize known stickers on reload |
| Modify | `tests/test_image_cache.py` | Test sticker recognition in history |
| Modify | `soul/instruction.md` | Add sticker behavioral guidelines |

---

### Task 1: StickerConfig

**Files:**
- Modify: `src/config.py:107-121`
- Modify: `config.example.toml`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sticker_store.py` with a config test:

```python
"""Sticker system tests."""

from src.config import BotConfig, StickerConfig


def test_sticker_config_defaults() -> None:
    cfg = StickerConfig()
    assert cfg.enabled is True
    assert cfg.storage_dir == "storage/stickers"
    assert cfg.max_count == 200


def test_bot_config_has_sticker() -> None:
    cfg = BotConfig()
    assert isinstance(cfg.sticker, StickerConfig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sticker_store.py::test_sticker_config_defaults tests/test_sticker_store.py::test_bot_config_has_sticker -v`
Expected: FAIL with `ImportError` — `StickerConfig` doesn't exist yet.

- [ ] **Step 3: Add StickerConfig to config.py**

Add after `VisionConfig` in `src/config.py`:

```python
class StickerConfig(BaseModel):
    """表情包系统配置。"""

    enabled: bool = True
    storage_dir: str = "storage/stickers"
    max_count: int = 200
```

And add to `BotConfig`:

```python
    sticker: StickerConfig = StickerConfig()
```

- [ ] **Step 4: Add [sticker] section to config.example.toml**

Add at end of `config.example.toml`:

```toml
# ---------------------------------------------------------------------------
# 表情包系统
# ---------------------------------------------------------------------------
[sticker]
# 总开关
enabled = true
# 表情包存储目录
storage_dir = "storage/stickers"
# 最大表情包数量（Dream 整理时执行）
max_count = 200
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_sticker_store.py -v`
Expected: PASS

- [ ] **Step 6: Run full lint + type check**

Run: `uv run ruff check src/config.py && uv run pyright src/config.py`

- [ ] **Step 7: Commit**

```bash
git add src/config.py config.example.toml tests/test_sticker_store.py
git commit -m "feat(sticker): add StickerConfig to BotConfig"
```

---

### Task 2: StickerStore core

**Files:**
- Create: `src/sticker/__init__.py`
- Create: `src/sticker/store.py`
- Test: `tests/test_sticker_store.py`

- [ ] **Step 1: Write failing tests for StickerStore**

Append to `tests/test_sticker_store.py`:

```python
import json
from pathlib import Path

import pytest

from src.sticker.store import StickerStore


@pytest.fixture
def store(tmp_path: Path) -> StickerStore:
    return StickerStore(storage_dir=str(tmp_path / "stickers"), max_count=10)


def test_init_creates_dir_and_empty_index(store: StickerStore) -> None:
    assert Path(store.storage_dir).exists()
    assert store.list_all() == {}


def test_format_prompt_view_empty(store: StickerStore) -> None:
    assert store.format_prompt_view() == "当前表情包库为空"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sticker_store.py::test_init_creates_dir_and_empty_index -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement StickerStore skeleton**

Create `src/sticker/__init__.py`:

```python
from .store import StickerStore

__all__ = ["StickerStore"]
```

Create `src/sticker/store.py`:

```python
"""Sticker store: flat file storage + JSON index for sticker management."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

_GIF87_MAGIC = b"GIF87a"
_GIF89_MAGIC = b"GIF89a"

# Map file signatures to extensions
_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"RIFF", "webp"),  # WebP starts with RIFF...WEBP but RIFF is enough + ext check
]


def _detect_ext(data: bytes) -> str | None:
    """Detect image format from file header. Returns extension or None."""
    for magic, ext in _SIGNATURES:
        if data[:len(magic)] == magic:
            return ext
    return None


def _is_gif(data: bytes) -> bool:
    return data[:6] in (_GIF87_MAGIC, _GIF89_MAGIC)


def _content_hash(data: bytes) -> str:
    """Compute short hash from image content (first 8 hex chars of SHA-256)."""
    return hashlib.sha256(data).hexdigest()[:8]


class StickerStore:
    """Manages sticker storage directory and JSON index."""

    def __init__(self, storage_dir: str, max_count: int = 200) -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_count = max_count
        self._index_path = self._dir / "index.json"
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    @property
    def storage_dir(self) -> str:
        return str(self._dir)

    @property
    def max_count(self) -> int:
        return self._max_count

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                self._index = data.get("stickers", {})
            except (json.JSONDecodeError, OSError):
                logger.warning("sticker index corrupted, starting fresh")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps({"stickers": self._index}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_all(self) -> dict[str, dict[str, Any]]:
        return dict(self._index)

    def get(self, sticker_id: str) -> dict[str, Any] | None:
        return self._index.get(sticker_id)

    def add(
        self,
        image_data: bytes,
        description: str,
        usage_hint: str,
        source: str = "auto",
    ) -> tuple[str, bool]:
        """Add a sticker from raw image data.

        Returns (sticker_id, is_new). If is_new is False, the sticker
        already existed (dedup by content hash).

        Raises ValueError if the image is a GIF or unrecognized format.
        """
        if _is_gif(image_data):
            raise ValueError("GIF 格式不支持，仅支持静态图片（JPG/PNG/WebP）")

        ext = _detect_ext(image_data)
        if ext is None:
            raise ValueError("无法识别的图片格式，仅支持 JPG/PNG/WebP")

        short_hash = _content_hash(image_data)
        sticker_id = f"stk_{short_hash}"

        if sticker_id in self._index:
            return sticker_id, False

        filename = f"{sticker_id}.{ext}"
        dest = self._dir / filename
        dest.write_bytes(image_data)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._index[sticker_id] = {
            "file": filename,
            "description": description,
            "usage_hint": usage_hint,
            "source": source,
            "send_count": 0,
            "last_sent": None,
            "created_at": now,
        }
        self._save_index()
        logger.info("sticker added | id={} source={}", sticker_id, source)
        return sticker_id, True

    def remove(self, sticker_id: str) -> bool:
        """Remove a sticker. Returns True if it existed."""
        entry = self._index.pop(sticker_id, None)
        if entry is None:
            return False
        file_path = self._dir / entry["file"]
        if file_path.exists():
            file_path.unlink()
        self._save_index()
        logger.info("sticker removed | id={}", sticker_id)
        return True

    def record_send(self, sticker_id: str) -> None:
        """Increment send_count and update last_sent."""
        entry = self._index.get(sticker_id)
        if entry is None:
            return
        entry["send_count"] = entry.get("send_count", 0) + 1
        entry["last_sent"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save_index()

    def resolve_path(self, sticker_id: str) -> Path | None:
        """Return the absolute file path for a sticker, or None."""
        entry = self._index.get(sticker_id)
        if entry is None:
            return None
        path = self._dir / entry["file"]
        return path if path.exists() else None

    def lookup_by_hash(self, image_data: bytes) -> str | None:
        """Check if image data matches a known sticker. Returns sticker_id or None."""
        short_hash = _content_hash(image_data)
        sticker_id = f"stk_{short_hash}"
        return sticker_id if sticker_id in self._index else None

    def format_prompt_view(self) -> str:
        """Format sticker index for system prompt injection.

        Only includes stable fields (id, description, usage_hint) to
        preserve cache stability. Volatile fields (send_count, last_sent,
        created_at) are excluded.
        """
        if not self._index:
            return "当前表情包库为空"
        lines = ["当前表情包库："]
        for stk_id, entry in self._index.items():
            desc = entry.get("description", "")
            hint = entry.get("usage_hint", "")
            lines.append(f"[表情包:{stk_id}] {desc} | {hint}")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sticker_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sticker/__init__.py src/sticker/store.py tests/test_sticker_store.py
git commit -m "feat(sticker): add StickerStore with index and storage management"
```

---

### Task 3: StickerStore full test coverage

**Files:**
- Test: `tests/test_sticker_store.py`

- [ ] **Step 1: Write comprehensive tests**

Append to `tests/test_sticker_store.py`:

```python
# -- Minimal valid images for testing --

# 1x1 JPEG (smallest valid JPEG)
_JPEG_1PX = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000"
    "ffdb004300080606070605080707070909080a0c"
    "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c"
    "20242e2720222c231c1c2837292c30313434341f"
    "27393d38323c2e333432ffc00011080001000103"
    "012200021101031101ffc4001f00000105010101"
    "01010100000000000000000102030405060708090a"
    "0bffc400b5100002010303020403050504040000"
    "017d01020300041105122131410613516107227114"
    "328191a1082342b1c11552d1f02433627282090a"
    "161718191a25262728292a3435363738393a434445"
    "464748494a535455565758595a636465666768696a"
    "737475767778797a838485868788898a9293949596"
    "9798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8"
    "b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9da"
    "e1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"
    "ffda000c03010002110311003f00fbfc0000000000"
    "00ffd9"
)

# Minimal PNG (1x1 white pixel)
_PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452"
    "00000001000000010802000000907753"
    "de0000000c4944415408d763f8cf0000"
    "0001010000186018660000000049454e"
    "44ae426082"
)

# GIF89a header (for rejection test)
_GIF_HEADER = b"GIF89a" + b"\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x00\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b"


def test_add_jpeg(store: StickerStore) -> None:
    stk_id, is_new = store.add(_JPEG_1PX, "test jpeg", "testing")
    assert is_new is True
    assert stk_id.startswith("stk_")
    assert store.get(stk_id) is not None
    assert store.get(stk_id)["description"] == "test jpeg"
    assert store.get(stk_id)["source"] == "auto"
    assert store.resolve_path(stk_id) is not None
    assert store.resolve_path(stk_id).exists()


def test_add_png(store: StickerStore) -> None:
    stk_id, is_new = store.add(_PNG_1PX, "test png", "testing")
    assert is_new is True
    assert stk_id.startswith("stk_")
    assert store.get(stk_id)["file"].endswith(".png")


def test_add_dedup(store: StickerStore) -> None:
    stk_id1, is_new1 = store.add(_JPEG_1PX, "first", "hint1")
    stk_id2, is_new2 = store.add(_JPEG_1PX, "second", "hint2")
    assert stk_id1 == stk_id2
    assert is_new1 is True
    assert is_new2 is False
    # Original description preserved
    assert store.get(stk_id1)["description"] == "first"


def test_add_gif_rejected(store: StickerStore) -> None:
    with pytest.raises(ValueError, match="GIF"):
        store.add(_GIF_HEADER, "gif sticker", "fun times")


def test_add_unknown_format_rejected(store: StickerStore) -> None:
    with pytest.raises(ValueError, match="无法识别"):
        store.add(b"not an image at all", "bad data", "hint")


def test_remove(store: StickerStore) -> None:
    stk_id, _ = store.add(_JPEG_1PX, "to remove", "hint")
    path = store.resolve_path(stk_id)
    assert path.exists()
    assert store.remove(stk_id) is True
    assert store.get(stk_id) is None
    assert not path.exists()


def test_remove_nonexistent(store: StickerStore) -> None:
    assert store.remove("stk_nonexistent") is False


def test_record_send(store: StickerStore) -> None:
    stk_id, _ = store.add(_JPEG_1PX, "sendable", "hint")
    assert store.get(stk_id)["send_count"] == 0
    assert store.get(stk_id)["last_sent"] is None
    store.record_send(stk_id)
    assert store.get(stk_id)["send_count"] == 1
    assert store.get(stk_id)["last_sent"] is not None
    store.record_send(stk_id)
    assert store.get(stk_id)["send_count"] == 2


def test_lookup_by_hash_found(store: StickerStore) -> None:
    stk_id, _ = store.add(_JPEG_1PX, "lookup test", "hint")
    assert store.lookup_by_hash(_JPEG_1PX) == stk_id


def test_lookup_by_hash_not_found(store: StickerStore) -> None:
    assert store.lookup_by_hash(_JPEG_1PX) is None


def test_format_prompt_view_with_stickers(store: StickerStore) -> None:
    store.add(_JPEG_1PX, "嫌弃猫", "对方无语时")
    store.add(_PNG_1PX, "熊猫头", "阴阳怪气时")
    view = store.format_prompt_view()
    assert "当前表情包库：" in view
    assert "嫌弃猫" in view
    assert "对方无语时" in view
    assert "熊猫头" in view
    # Volatile fields must NOT appear
    assert "send_count" not in view
    assert "last_sent" not in view
    assert "created_at" not in view


def test_add_source_admin(store: StickerStore) -> None:
    stk_id, _ = store.add(_JPEG_1PX, "admin fed", "hint", source="admin")
    assert store.get(stk_id)["source"] == "admin"


def test_index_persists_across_reload(tmp_path: Path) -> None:
    store1 = StickerStore(storage_dir=str(tmp_path / "stickers"), max_count=10)
    stk_id, _ = store1.add(_JPEG_1PX, "persist test", "hint")

    store2 = StickerStore(storage_dir=str(tmp_path / "stickers"), max_count=10)
    assert store2.get(stk_id) is not None
    assert store2.get(stk_id)["description"] == "persist test"
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/test_sticker_store.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_sticker_store.py
git commit -m "test(sticker): comprehensive StickerStore tests"
```

---

### Task 4: Sticker conversation tools

**Files:**
- Create: `src/tools/sticker_tools.py`
- Create: `tests/test_sticker_tools.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sticker_tools.py`:

```python
"""Sticker tool tests."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.sticker.store import StickerStore
from src.tools.context import ToolContext
from src.tools.sticker_tools import SaveStickerTool, SendStickerTool

# Minimal JPEG for testing
_JPEG_1PX = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000"
    "ffdb004300080606070605080707070909080a0c"
    "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c"
    "20242e2720222c231c1c2837292c30313434341f"
    "27393d38323c2e333432ffc00011080001000103"
    "012200021101031101ffc4001f00000105010101"
    "01010100000000000000000102030405060708090a"
    "0bffc400b5100002010303020403050504040000"
    "017d01020300041105122131410613516107227114"
    "328191a1082342b1c11552d1f02433627282090a"
    "161718191a25262728292a3435363738393a434445"
    "464748494a535455565758595a636465666768696a"
    "737475767778797a838485868788898a9293949596"
    "9798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8"
    "b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9da"
    "e1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"
    "ffda000c03010002110311003f00fbfc0000000000"
    "00ffd9"
)

_GIF_DATA = b"GIF89a" + b"\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x00\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b"


@pytest.fixture
def sticker_store(tmp_path: Path) -> StickerStore:
    return StickerStore(storage_dir=str(tmp_path / "stickers"), max_count=10)


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    d = tmp_path / "image_cache" / "ab"
    d.mkdir(parents=True)
    return d


# ── SaveStickerTool ──


async def test_save_sticker_success(
    sticker_store: StickerStore, image_dir: Path,
) -> None:
    img_path = image_dir / "abc123.jpg"
    img_path.write_bytes(_JPEG_1PX)

    tool = SaveStickerTool(sticker_store, superusers={"admin1"})
    ctx = ToolContext(user_id="user1", group_id="100")
    result = await tool.execute(
        ctx,
        image_ref=str(img_path),
        description="嫌弃猫",
        usage_hint="无语时",
    )
    assert "已收录" in result
    assert "stk_" in result


async def test_save_sticker_dedup(
    sticker_store: StickerStore, image_dir: Path,
) -> None:
    img_path = image_dir / "abc123.jpg"
    img_path.write_bytes(_JPEG_1PX)

    tool = SaveStickerTool(sticker_store, superusers=set())
    ctx = ToolContext(user_id="user1", group_id="100")
    await tool.execute(ctx, image_ref=str(img_path), description="first", usage_hint="hint")
    result = await tool.execute(ctx, image_ref=str(img_path), description="second", usage_hint="hint")
    assert "已存在" in result


async def test_save_sticker_gif_rejected(
    sticker_store: StickerStore, image_dir: Path,
) -> None:
    gif_path = image_dir / "anim.gif"
    gif_path.write_bytes(_GIF_DATA)

    tool = SaveStickerTool(sticker_store, superusers=set())
    ctx = ToolContext(user_id="user1", group_id="100")
    result = await tool.execute(
        ctx, image_ref=str(gif_path), description="gif", usage_hint="hint",
    )
    assert "GIF" in result


async def test_save_sticker_missing_file(sticker_store: StickerStore) -> None:
    tool = SaveStickerTool(sticker_store, superusers=set())
    ctx = ToolContext(user_id="user1", group_id="100")
    result = await tool.execute(
        ctx, image_ref="/nonexistent/path.jpg", description="x", usage_hint="y",
    )
    assert "找不到" in result


async def test_save_sticker_admin_source(
    sticker_store: StickerStore, image_dir: Path,
) -> None:
    img_path = image_dir / "abc123.jpg"
    img_path.write_bytes(_JPEG_1PX)

    tool = SaveStickerTool(sticker_store, superusers={"admin1"})
    ctx = ToolContext(user_id="admin1", group_id="100")
    result = await tool.execute(
        ctx, image_ref=str(img_path), description="admin sticker", usage_hint="hint",
    )
    assert "已收录" in result
    # Verify source is admin
    stk_id = result.split("已收录")[0].strip().split()[-1]
    # Find the sticker by checking the store
    all_stickers = sticker_store.list_all()
    admin_sticker = next(s for s in all_stickers.values() if s["description"] == "admin sticker")
    assert admin_sticker["source"] == "admin"


# ── SendStickerTool ──


async def test_send_sticker_success(
    sticker_store: StickerStore,
) -> None:
    stk_id, _ = sticker_store.add(_JPEG_1PX, "cat", "fun")

    mock_bot = AsyncMock()
    tool = SendStickerTool(sticker_store)
    ctx = ToolContext(bot=mock_bot, user_id="user1", group_id="100")
    result = await tool.execute(ctx, sticker_id=stk_id)
    assert "已发送" in result
    mock_bot.send_group_msg.assert_called_once()
    assert sticker_store.get(stk_id)["send_count"] == 1


async def test_send_sticker_private(
    sticker_store: StickerStore,
) -> None:
    stk_id, _ = sticker_store.add(_JPEG_1PX, "cat", "fun")

    mock_bot = AsyncMock()
    tool = SendStickerTool(sticker_store)
    ctx = ToolContext(bot=mock_bot, user_id="user1", group_id=None)
    result = await tool.execute(ctx, sticker_id=stk_id)
    assert "已发送" in result
    mock_bot.send_private_msg.assert_called_once()


async def test_send_sticker_not_found(sticker_store: StickerStore) -> None:
    tool = SendStickerTool(sticker_store)
    ctx = ToolContext(bot=AsyncMock(), user_id="user1", group_id="100")
    result = await tool.execute(ctx, sticker_id="stk_nonexistent")
    assert "不存在" in result


async def test_send_sticker_no_bot(sticker_store: StickerStore) -> None:
    stk_id, _ = sticker_store.add(_JPEG_1PX, "cat", "fun")
    tool = SendStickerTool(sticker_store)
    ctx = ToolContext(bot=None, user_id="user1", group_id="100")
    result = await tool.execute(ctx, sticker_id=stk_id)
    assert "不可用" in result


def test_save_tool_schema() -> None:
    tool = SaveStickerTool(StickerStore("/tmp/test_stk", max_count=5), superusers=set())
    schema = tool.parameters
    assert "image_ref" in schema["properties"]
    assert "description" in schema["properties"]
    assert "usage_hint" in schema["properties"]


def test_send_tool_schema() -> None:
    tool = SendStickerTool(StickerStore("/tmp/test_stk", max_count=5))
    schema = tool.parameters
    assert "sticker_id" in schema["properties"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sticker_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement sticker tools**

Create `src/tools/sticker_tools.py`:

```python
"""Sticker tools: save and send stickers during conversation."""

from pathlib import Path
from typing import Any

from loguru import logger

from src.sticker.store import StickerStore
from src.tools.base import Tool
from src.tools.context import ToolContext


class SaveStickerTool(Tool):
    def __init__(self, store: StickerStore, superusers: set[str]) -> None:
        self._store = store
        self._superusers = superusers

    @property
    def name(self) -> str:
        return "save_sticker"

    @property
    def description(self) -> str:
        return (
            "收录一张表情包到你的表情包库。"
            "只在你完全理解图片含义、清楚使用场景、且符合自己性格时才调用。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_ref": {
                    "type": "string",
                    "description": "当前对话中图片的磁盘路径（从 image_ref 块获取）",
                },
                "description": {
                    "type": "string",
                    "description": "图片内容描述",
                },
                "usage_hint": {
                    "type": "string",
                    "description": "什么时候适合发这张表情包",
                },
            },
            "required": ["image_ref", "description", "usage_hint"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        image_ref: str = kwargs["image_ref"]
        description: str = kwargs["description"]
        usage_hint: str = kwargs["usage_hint"]

        path = Path(image_ref)
        if not path.exists():
            return f"找不到图片文件: {image_ref}"

        image_data = path.read_bytes()
        source = "admin" if ctx.user_id in self._superusers else "auto"

        try:
            stk_id, is_new = self._store.add(image_data, description, usage_hint, source=source)
        except ValueError as e:
            return str(e)

        if not is_new:
            return f"表情包已存在: {stk_id}"

        return f"{stk_id} 已收录"


class SendStickerTool(Tool):
    def __init__(self, store: StickerStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "send_sticker"

    @property
    def description(self) -> str:
        return "发送一张表情包（作为单独的图片消息）。从表情包库中选择合适的表情包发送。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sticker_id": {
                    "type": "string",
                    "description": "表情包 ID，如 stk_a1b2c3d4（从表情包库列表获取）",
                },
            },
            "required": ["sticker_id"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        sticker_id: str = kwargs["sticker_id"]

        if not ctx.bot:
            return "Bot 不可用"

        file_path = self._store.resolve_path(sticker_id)
        if file_path is None:
            return f"表情包不存在: {sticker_id}"

        from nonebot.adapters.onebot.v11 import MessageSegment

        img_seg = MessageSegment.image(file_path)

        try:
            if ctx.group_id:
                await ctx.bot.send_group_msg(
                    group_id=int(ctx.group_id), message=img_seg,
                )
            else:
                await ctx.bot.send_private_msg(
                    user_id=int(ctx.user_id), message=img_seg,
                )
        except Exception:
            logger.exception("send_sticker failed | id={}", sticker_id)
            return f"发送失败: {sticker_id}"

        self._store.record_send(sticker_id)
        return f"已发送表情包 {sticker_id}"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_sticker_tools.py -v`
Expected: PASS

- [ ] **Step 5: Run lint + type check**

Run: `uv run ruff check src/tools/sticker_tools.py src/sticker/store.py && uv run pyright src/tools/sticker_tools.py src/sticker/store.py`

- [ ] **Step 6: Commit**

```bash
git add src/tools/sticker_tools.py tests/test_sticker_tools.py
git commit -m "feat(sticker): add save_sticker and send_sticker tools"
```

---

### Task 5: Prompt injection — sticker index in entity block

**Files:**
- Modify: `src/llm/prompt.py:66-88`
- Modify: `tests/test_prompt.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_prompt.py`:

```python
from src.sticker.store import StickerStore


@pytest.fixture
def sticker_store(tmp_path) -> StickerStore:
    return StickerStore(storage_dir=str(tmp_path / "stickers"), max_count=10)


async def test_build_blocks_includes_sticker_index(
    identity: Identity, store: MemoStore, sticker_store: StickerStore,
) -> None:
    # Minimal JPEG
    jpeg = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000"
        "ffdb004300080606070605080707070909080a0c"
        "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c"
        "20242e2720222c231c1c2837292c30313434341f"
        "27393d38323c2e333432ffc00011080001000103"
        "012200021101031101ffc4001f00000105010101"
        "01010100000000000000000102030405060708090a"
        "0bffc400b5100002010303020403050504040000"
        "017d01020300041105122131410613516107227114"
        "328191a1082342b1c11552d1f02433627282090a"
        "161718191a25262728292a3435363738393a434445"
        "464748494a535455565758595a636465666768696a"
        "737475767778797a838485868788898a9293949596"
        "9798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8"
        "b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9da"
        "e1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"
        "ffda000c03010002110311003f00fbfc0000000000"
        "00ffd9"
    )
    sticker_store.add(jpeg, "嫌弃猫", "对方无语时")

    pb = PromptBuilder(instruction="", sticker_store=sticker_store)
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="100", group_id=None, memo_store=store)
    entity_text = blocks[1]["text"]
    assert "[表情包:stk_" in entity_text
    assert "嫌弃猫" in entity_text
    assert "对方无语时" in entity_text


async def test_build_blocks_empty_sticker_store(
    identity: Identity, store: MemoStore, sticker_store: StickerStore,
) -> None:
    pb = PromptBuilder(instruction="", sticker_store=sticker_store)
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="100", group_id=None, memo_store=store)
    entity_text = blocks[1]["text"]
    assert "当前表情包库为空" in entity_text


async def test_build_blocks_no_sticker_store(
    identity: Identity, store: MemoStore,
) -> None:
    pb = PromptBuilder(instruction="")
    pb.build_static(identity, bot_self_id="999")
    blocks = await pb.build_blocks(user_id="100", group_id=None, memo_store=store)
    entity_text = blocks[1]["text"]
    assert "表情包" not in entity_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt.py::test_build_blocks_includes_sticker_index -v`
Expected: FAIL — `PromptBuilder` doesn't accept `sticker_store` yet.

- [ ] **Step 3: Modify PromptBuilder**

In `src/llm/prompt.py`, update `__init__` and `build_blocks`:

```python
from src.sticker.store import StickerStore

class PromptBuilder:
    def __init__(
        self,
        instruction: str = "",
        admins: dict[str, str] | None = None,
        sticker_store: StickerStore | None = None,
    ) -> None:
        self._instruction = instruction
        self._admins = admins or {}
        self._sticker_store = sticker_store
        self._static_block: dict[str, Any] = {}
```

In `build_blocks`, append sticker index to the entity text before creating the block:

```python
    async def build_blocks(
        self,
        user_id: str,
        group_id: str | None,
        memo_store: MemoStore,
    ) -> list[dict[str, Any]]:
        """Returns [static_block, entity_block]. Called per chat()."""
        text = f"【全局索引】\n{memo_store.serialize_index()}"
        if group_id:
            memo = memo_store.read(f"group_{group_id}")
            body = memo.body if memo else ""
            text += f"\n\n【当前在群 #{group_id} 中对话】\n{body}"
        else:
            memo = memo_store.read(f"user_{user_id}")
            body = memo.body if memo else ""
            text += f"\n\n【当前私聊 @{user_id}】\n{body}"

        if self._sticker_store:
            text += f"\n\n{self._sticker_store.format_prompt_view()}"

        entity_block = {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
        return [self._static_block, entity_block]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: PASS (including existing tests — `sticker_store` defaults to `None`).

- [ ] **Step 5: Commit**

```bash
git add src/llm/prompt.py tests/test_prompt.py
git commit -m "feat(sticker): inject sticker index into system prompt entity block"
```

---

### Task 6: Register tools and wire StickerStore in plugin init

**Files:**
- Modify: `src/plugins/chat/__init__.py`

- [ ] **Step 1: Add StickerStore initialization and tool registration**

In the `_startup` function (around line 86), add after `_image_cache` setup:

```python
from src.sticker.store import StickerStore
from src.tools.sticker_tools import SaveStickerTool, SendStickerTool
```

After the vision config block (~line 76), add:

```python
    _sticker_store: StickerStore | None = None
    if bot_config.sticker.enabled:
        _sticker_store = StickerStore(
            storage_dir=bot_config.sticker.storage_dir,
            max_count=bot_config.sticker.max_count,
        )
```

After existing tool registrations (~line 102), add:

```python
    if _sticker_store is not None:
        tools.register(SaveStickerTool(_sticker_store, superusers))
        tools.register(SendStickerTool(_sticker_store))
```

Update `PromptBuilder` construction (~line 109) to pass `sticker_store`:

```python
    prompt_builder = PromptBuilder(
        instruction=instruction, admins=bot_config.admins,
        sticker_store=_sticker_store,
    )
```

- [ ] **Step 2: Run lint + type check**

Run: `uv run ruff check src/plugins/chat/__init__.py && uv run pyright src/plugins/chat/__init__.py`

- [ ] **Step 3: Commit**

```bash
git add src/plugins/chat/__init__.py
git commit -m "feat(sticker): register sticker tools and wire StickerStore"
```

---

### Task 7: History reload — sticker recognition

**Files:**
- Modify: `src/memory/history_loader.py`
- Test: `tests/test_image_cache.py` (or new test file)

- [ ] **Step 1: Write failing test**

Add a test file `tests/test_history_sticker.py`:

```python
"""Test sticker recognition during history reload."""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.group_timeline import GroupTimeline
from src.memory.history_loader import _extract_content
from src.memory.image_cache import ImageCache
from src.memory.types import ImageRefBlock
from src.sticker.store import StickerStore

# Minimal JPEG
_JPEG_1PX = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000"
    "ffdb004300080606070605080707070909080a0c"
    "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c"
    "20242e2720222c231c1c2837292c30313434341f"
    "27393d38323c2e333432ffc00011080001000103"
    "012200021101031101ffc4001f00000105010101"
    "01010100000000000000000102030405060708090a"
    "0bffc400b5100002010303020403050504040000"
    "017d01020300041105122131410613516107227114"
    "328191a1082342b1c11552d1f02433627282090a"
    "161718191a25262728292a3435363738393a434445"
    "464748494a535455565758595a636465666768696a"
    "737475767778797a838485868788898a9293949596"
    "9798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8"
    "b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9da"
    "e1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"
    "ffda000c03010002110311003f00fbfc0000000000"
    "00ffd9"
)


async def test_extract_content_recognizes_sticker(tmp_path: Path) -> None:
    """When a downloaded image matches a known sticker, use sticker path."""
    sticker_store = StickerStore(str(tmp_path / "stickers"), max_count=10)
    stk_id, _ = sticker_store.add(_JPEG_1PX, "test cat", "when bored")
    sticker_path = sticker_store.resolve_path(stk_id)

    image_cache = ImageCache(tmp_path / "image_cache")

    # Mock session that returns the same JPEG data
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.read = AsyncMock(return_value=_JPEG_1PX)
    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_response),
        __aexit__=AsyncMock(return_value=False),
    ))

    segments = [
        {"type": "image", "data": {"url": "https://example.com/img.jpg", "file": "abc123.jpg"}},
    ]

    content = await _extract_content(
        segments, mock_session, image_cache, sticker_store=sticker_store,
    )

    # Should be a list with an ImageRefBlock pointing to sticker storage
    assert isinstance(content, list)
    img_block = next(b for b in content if b["type"] == "image_ref")
    assert str(sticker_path) in img_block["path"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_history_sticker.py -v`
Expected: FAIL — `_extract_content` doesn't accept `sticker_store` param.

- [ ] **Step 3: Modify history_loader.py**

Update `_extract_content` to accept an optional `StickerStore` and check downloaded images:

```python
from src.sticker.store import StickerStore

async def _extract_content(
    segments: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    image_cache: ImageCache | None,
    sticker_store: StickerStore | None = None,
) -> Content:
```

In the image download resolution section (after `asyncio.gather`), add sticker recognition. Replace the current image result processing with:

```python
    # Resolve all image downloads concurrently
    images: list[ImageRefBlock] = []
    if image_tasks:
        t0 = time.perf_counter()
        results = await asyncio.gather(*image_tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException) or r is None:
                text_parts.append("[图片]")
            elif sticker_store is not None:
                # Check if downloaded image matches a known sticker
                img_path = Path(r["path"])
                if img_path.exists():
                    img_data = img_path.read_bytes()
                    stk_id = sticker_store.lookup_by_hash(img_data)
                    if stk_id is not None:
                        stk_path = sticker_store.resolve_path(stk_id)
                        if stk_path is not None:
                            images.append(ImageRefBlock(
                                type="image_ref", path=str(stk_path), media_type=r["media_type"],
                            ))
                            # Remove duplicate from image_cache
                            img_path.unlink(missing_ok=True)
                            continue
                images.append(r)
            else:
                images.append(r)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "history image batch | tasks={} ok={} elapsed={:.0f}ms",
            len(image_tasks), len(images), elapsed_ms,
        )
```

Also update `_load_one_group` to pass `sticker_store`:

```python
async def _load_one_group(
    session: aiohttp.ClientSession,
    napcat_url: str,
    group_id: str,
    timeline: GroupTimeline,
    count: int,
    bot_self_id: str = "",
    image_cache: ImageCache | None = None,
    sticker_store: StickerStore | None = None,
) -> None:
```

And `load_group_history`:

```python
async def load_group_history(
    napcat_url: str,
    group_ids: list[str],
    timeline: GroupTimeline,
    count: int = 30,
    bot_self_id: str = "",
    image_cache: ImageCache | None = None,
    sticker_store: StickerStore | None = None,
) -> None:
```

Pass `sticker_store` down through the call chain.

- [ ] **Step 4: Update load_group_history call in chat/__init__.py**

In `_on_connect` (~line 202), pass `sticker_store`:

```python
        await load_group_history(
            napcat_url=bot_config.napcat.api_url,
            group_ids=group_ids,
            timeline=_timeline,
            count=bot_config.group.history_load_count,
            bot_self_id=bot.self_id,
            image_cache=_image_cache if _vision_enabled else None,
            sticker_store=_sticker_store,
        )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_history_sticker.py -v`
Expected: PASS

- [ ] **Step 6: Run existing tests to verify no regression**

Run: `uv run pytest tests/ -v`
Expected: PASS (existing history_loader tests should still work since `sticker_store` defaults to `None`).

- [ ] **Step 7: Commit**

```bash
git add src/memory/history_loader.py tests/test_history_sticker.py src/plugins/chat/__init__.py
git commit -m "feat(sticker): recognize known stickers during history reload"
```

---

### Task 8: Dream integration — sticker cleanup tools

**Files:**
- Modify: `src/llm/dream.py`
- Modify: `tests/test_dream.py`

- [ ] **Step 1: Read existing dream tests for patterns**

Check `tests/test_dream.py` for the testing style used.

- [ ] **Step 2: Write failing tests**

Add to `tests/test_dream.py`:

```python
from src.sticker.store import StickerStore

# Minimal JPEG (reuse from other test files)
_JPEG_1PX = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000"
    "ffdb004300080606070605080707070909080a0c"
    "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c"
    "20242e2720222c231c1c2837292c30313434341f"
    "27393d38323c2e333432ffc00011080001000103"
    "012200021101031101ffc4001f00000105010101"
    "01010100000000000000000102030405060708090a"
    "0bffc400b5100002010303020403050504040000"
    "017d01020300041105122131410613516107227114"
    "328191a1082342b1c11552d1f02433627282090a"
    "161718191a25262728292a3435363738393a434445"
    "464748494a535455565758595a636465666768696a"
    "737475767778797a838485868788898a9293949596"
    "9798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8"
    "b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9da"
    "e1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa"
    "ffda000c03010002110311003f00fbfc0000000000"
    "00ffd9"
)


async def test_dream_list_stickers(tmp_path) -> None:
    sticker_store = StickerStore(str(tmp_path / "stickers"), max_count=10)
    sticker_store.add(_JPEG_1PX, "test cat", "bored")

    agent = DreamAgent(store=MemoStore(base_dir=str(tmp_path / "memo")), sticker_store=sticker_store)
    result = await agent._execute_tool("list_stickers", {})
    assert "test cat" in result
    assert "stk_" in result


async def test_dream_delete_sticker(tmp_path) -> None:
    sticker_store = StickerStore(str(tmp_path / "stickers"), max_count=10)
    stk_id, _ = sticker_store.add(_JPEG_1PX, "to delete", "hint")

    agent = DreamAgent(store=MemoStore(base_dir=str(tmp_path / "memo")), sticker_store=sticker_store)
    result = await agent._execute_tool("delete_sticker", {"id": stk_id})
    assert "已删除" in result
    assert sticker_store.get(stk_id) is None


async def test_dream_delete_sticker_not_found(tmp_path) -> None:
    sticker_store = StickerStore(str(tmp_path / "stickers"), max_count=10)
    agent = DreamAgent(store=MemoStore(base_dir=str(tmp_path / "memo")), sticker_store=sticker_store)
    result = await agent._execute_tool("delete_sticker", {"id": "stk_nonexist"})
    assert "未找到" in result
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_dream.py::test_dream_list_stickers -v`
Expected: FAIL — `DreamAgent` doesn't accept `sticker_store`.

- [ ] **Step 4: Modify DreamAgent**

In `src/llm/dream.py`, add sticker tools:

Add tool definitions after `_UPDATE_MEMO_TOOL`:

```python
_LIST_STICKERS_TOOL: dict[str, Any] = {
    "name": "list_stickers",
    "description": "查看当前表情包库的完整索引（含使用统计）。",
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}

_DELETE_STICKER_TOOL: dict[str, Any] = {
    "name": "delete_sticker",
    "description": "删除一张表情包（文件和索引同时清除）。",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "表情包 ID，如 stk_a1b2c3d4",
            },
        },
        "required": ["id"],
    },
}
```

Update `DreamAgent.__init__` to accept `sticker_store`:

```python
from src.sticker.store import StickerStore

class DreamAgent:
    def __init__(
        self,
        store: MemoStore,
        interval_hours: int = 24,
        min_compacts: int = 5,
        max_rounds: int = 15,
        user_max_chars: int = 300,
        group_max_chars: int = 500,
        sticker_store: StickerStore | None = None,
    ) -> None:
        # ... existing fields ...
        self._sticker_store = sticker_store
```

Update `_run` to include sticker tools and prompt:

In the tools list:
```python
            tools = [_RECALL_MEMO_TOOL, _UPDATE_MEMO_TOOL]
            if self._sticker_store:
                tools.extend([_LIST_STICKERS_TOOL, _DELETE_STICKER_TOOL])
```

In the system prompt, add after the existing instructions (before the closing template section):

```python
            sticker_section = ""
            if self._sticker_store:
                sticker_section = (
                    "\n\n4. 表情包库整理：用 list_stickers 查看完整索引，"
                    "审查 send_count 低且 created_at 久远的表情包（LRU 候选），"
                    "综合判断是否淘汰（独特/有价值的可以保留），"
                    f"用 delete_sticker 删除不需要的。库存上限 {self._sticker_store.max_count} 张。"
                    "同时检查 description 和 usage_hint 是否准确，如有需要可以更新。"
                )
```

Update `_execute_tool` to handle sticker tools:

```python
        if name == "list_stickers":
            if self._sticker_store is None:
                return "表情包系统未启用"
            import json
            return json.dumps(self._sticker_store.list_all(), ensure_ascii=False, indent=2)

        if name == "delete_sticker":
            if self._sticker_store is None:
                return "表情包系统未启用"
            sticker_id = input.get("id", "")
            if not sticker_id:
                return "缺少 id 参数"
            if self._sticker_store.remove(sticker_id):
                return f"已删除: {sticker_id}"
            return f"未找到: {sticker_id}"
```

- [ ] **Step 5: Update DreamAgent construction in chat/__init__.py**

In `_startup` (~line 113):

```python
    _dream = DreamAgent(
        store=memo_store,
        interval_hours=bot_config.dream.interval_hours,
        min_compacts=bot_config.dream.min_compacts,
        max_rounds=bot_config.dream.max_rounds,
        user_max_chars=bot_config.memo.user_max_chars,
        group_max_chars=bot_config.memo.group_max_chars,
        sticker_store=_sticker_store,
    )
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_dream.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/llm/dream.py src/plugins/chat/__init__.py tests/test_dream.py
git commit -m "feat(sticker): add sticker cleanup tools to Dream agent"
```

---

### Task 9: Soul instructions — sticker behavioral guidelines

**Files:**
- Modify: `soul/instruction.md`

- [ ] **Step 1: Add sticker section to instruction.md**

Add at the end of `soul/instruction.md`:

```markdown
## 表情包

你有一个表情包库，可以收录和发送表情包。

### 收录原则
- 只收录你**完全理解**含义和情绪的表情包
- 要能明确说出什么场景会用它
- 符合你的性格设定，你以后确实会想发
- 不要为了收藏而收藏
- GIF 动图不支持

### 发送原则
- 自然融入对话节奏，就像真人一样偶尔甩个表情包
- 不要刻意或过度使用
- 表情包会作为独立的图片消息发送，不会和文字混在一起
```

- [ ] **Step 2: Commit**

```bash
git add soul/instruction.md
git commit -m "feat(sticker): add sticker behavioral guidelines to soul instructions"
```

---

### Task 10: Full integration test + final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run lint**

Run: `uv run ruff check src/`

- [ ] **Step 3: Run type check**

Run: `uv run pyright`

- [ ] **Step 4: Verify sticker directory structure**

Check that the `storage/stickers/` directory is set up to be created at runtime (not committed to git). If `.gitignore` exists, verify `storage/` is already ignored.

- [ ] **Step 5: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix: address lint/type issues from sticker system integration"
```
