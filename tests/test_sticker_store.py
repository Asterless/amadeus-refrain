"""Tests for StickerConfig and StickerStore."""

import pytest

from src.config import BotConfig, StickerConfig
from src.sticker.store import StickerStore

# ---------------------------------------------------------------------------
# Minimal valid image byte sequences (magic bytes + padding for distinct hashes)
# ---------------------------------------------------------------------------

# JPEG: starts with ff d8 ff
_JPEG_DATA = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"jpeg-payload-a"

# PNG: starts with 89 50 4e 47 0d 0a 1a 0a
_PNG_DATA = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64 + b"png-payload-a"

# Another JPEG with different content for dedup testing
_JPEG_DATA_B = b"\xff\xd8\xff\xe0" + b"\x01" * 64 + b"jpeg-payload-b"

# WebP: RIFF....WEBP
_WEBP_DATA = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 64 + b"webp-payload"

# GIF: starts with GIF89a
_GIF_DATA = b"GIF89a" + b"\x00" * 64 + b"gif-payload"

# GIF87a variant
_GIF87_DATA = b"GIF87a" + b"\x00" * 64 + b"gif87-payload"

# Unknown format
_UNKNOWN_DATA = b"\x00\x01\x02\x03" + b"\xde\xad\xbe\xef" * 20


# ---------------------------------------------------------------------------
# Task 1: StickerConfig
# ---------------------------------------------------------------------------


def test_sticker_config_defaults() -> None:
    cfg = StickerConfig()
    assert cfg.enabled is True
    assert cfg.storage_dir == "storage/stickers"
    assert cfg.max_count == 200


def test_sticker_config_custom() -> None:
    cfg = StickerConfig(enabled=False, storage_dir="/tmp/stickers", max_count=50)
    assert cfg.enabled is False
    assert cfg.storage_dir == "/tmp/stickers"
    assert cfg.max_count == 50


def test_bot_config_has_sticker_field() -> None:
    bot = BotConfig()
    assert hasattr(bot, "sticker")
    assert isinstance(bot.sticker, StickerConfig)


def test_bot_config_sticker_defaults() -> None:
    bot = BotConfig()
    assert bot.sticker.enabled is True
    assert bot.sticker.storage_dir == "storage/stickers"
    assert bot.sticker.max_count == 200


def test_bot_config_sticker_override() -> None:
    bot = BotConfig(sticker=StickerConfig(max_count=100))
    assert bot.sticker.max_count == 100


# ---------------------------------------------------------------------------
# Task 2 + 3: StickerStore
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> StickerStore:
    return StickerStore(storage_dir=str(tmp_path / "stickers"))


# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------


def test_init_creates_directory(tmp_path) -> None:
    path = tmp_path / "new_stickers"
    assert not path.exists()
    StickerStore(storage_dir=str(path))
    assert path.is_dir()


def test_init_empty_index(store: StickerStore) -> None:
    assert store.list_all() == {}


def test_storage_dir_property(tmp_path) -> None:
    path = tmp_path / "stickers"
    s = StickerStore(storage_dir=str(path))
    assert s.storage_dir == path


def test_max_count_property(tmp_path) -> None:
    s = StickerStore(storage_dir=str(tmp_path / "s"), max_count=42)
    assert s.max_count == 42


# ------------------------------------------------------------------
# add() — success cases
# ------------------------------------------------------------------


def test_add_jpeg(store: StickerStore) -> None:
    sid, is_new = store.add(_JPEG_DATA, "测试JPEG", "开心时发", source="auto")
    assert sid.startswith("stk_")
    assert is_new is True
    entry = store.get(sid)
    assert entry is not None
    assert entry["description"] == "测试JPEG"
    assert entry["usage_hint"] == "开心时发"
    assert entry["source"] == "auto"
    assert entry["send_count"] == 0
    assert entry["last_sent"] is None
    assert "created_at" in entry
    assert entry["filename"].endswith(".jpg")


def test_add_png(store: StickerStore) -> None:
    sid, is_new = store.add(_PNG_DATA, "测试PNG", "悲伤时发", source="admin")
    assert sid.startswith("stk_")
    assert is_new is True
    entry = store.get(sid)
    assert entry is not None
    assert entry["filename"].endswith(".png")
    assert entry["source"] == "admin"


def test_add_webp(store: StickerStore) -> None:
    sid, is_new = store.add(_WEBP_DATA, "测试WebP", "任何时候", source="auto")
    assert sid.startswith("stk_")
    assert is_new is True
    entry = store.get(sid)
    assert entry is not None
    assert entry["filename"].endswith(".webp")


def test_add_writes_file(store: StickerStore) -> None:
    sid, _ = store.add(_JPEG_DATA, "desc", "hint")
    path = store.resolve_path(sid)
    assert path is not None
    assert path.exists()
    assert path.read_bytes() == _JPEG_DATA


def test_add_source_admin(store: StickerStore) -> None:
    sid, is_new = store.add(_JPEG_DATA, "管理员添加", "正式使用", source="admin")
    assert is_new is True
    assert store.get(sid)["source"] == "admin"


# ------------------------------------------------------------------
# add() — deduplication
# ------------------------------------------------------------------


def test_add_dedup_same_image(store: StickerStore) -> None:
    sid1, is_new1 = store.add(_JPEG_DATA, "first", "hint1")
    sid2, is_new2 = store.add(_JPEG_DATA, "second", "hint2")
    assert sid1 == sid2
    assert is_new1 is True
    assert is_new2 is False
    # Metadata should still be from the first add
    assert store.get(sid1)["description"] == "first"


def test_add_two_different_images_are_distinct(store: StickerStore) -> None:
    sid1, _ = store.add(_JPEG_DATA, "a", "hint")
    sid2, _ = store.add(_JPEG_DATA_B, "b", "hint")
    assert sid1 != sid2
    assert len(store.list_all()) == 2


# ------------------------------------------------------------------
# add() — rejection
# ------------------------------------------------------------------


def test_add_gif_rejected(store: StickerStore) -> None:
    with pytest.raises(ValueError, match="GIF"):
        store.add(_GIF_DATA, "gif", "hint")


def test_add_gif87_rejected(store: StickerStore) -> None:
    with pytest.raises(ValueError, match="GIF"):
        store.add(_GIF87_DATA, "gif87", "hint")


def test_add_unknown_format_rejected(store: StickerStore) -> None:
    with pytest.raises(ValueError):
        store.add(_UNKNOWN_DATA, "unknown", "hint")


# ------------------------------------------------------------------
# remove()
# ------------------------------------------------------------------


def test_remove_existing(store: StickerStore) -> None:
    sid, _ = store.add(_JPEG_DATA, "desc", "hint")
    path = store.resolve_path(sid)
    assert path is not None and path.exists()

    result = store.remove(sid)
    assert result is True
    assert store.get(sid) is None
    assert not path.exists()


def test_remove_nonexistent(store: StickerStore) -> None:
    result = store.remove("stk_00000000")
    assert result is False


def test_remove_updates_list(store: StickerStore) -> None:
    sid, _ = store.add(_JPEG_DATA, "desc", "hint")
    store.add(_PNG_DATA, "png", "hint")
    store.remove(sid)
    assert sid not in store.list_all()
    assert len(store.list_all()) == 1


# ------------------------------------------------------------------
# record_send()
# ------------------------------------------------------------------


def test_record_send_increments_count(store: StickerStore) -> None:
    sid, _ = store.add(_JPEG_DATA, "desc", "hint")
    assert store.get(sid)["send_count"] == 0

    store.record_send(sid)
    assert store.get(sid)["send_count"] == 1

    store.record_send(sid)
    assert store.get(sid)["send_count"] == 2


def test_record_send_updates_last_sent(store: StickerStore) -> None:
    sid, _ = store.add(_JPEG_DATA, "desc", "hint")
    assert store.get(sid)["last_sent"] is None

    store.record_send(sid)
    last_sent = store.get(sid)["last_sent"]
    assert last_sent is not None
    # Should be a valid ISO datetime string
    from datetime import datetime
    dt = datetime.fromisoformat(last_sent)
    assert dt.tzinfo is not None  # timezone-aware


def test_record_send_nonexistent_is_noop(store: StickerStore) -> None:
    # Should not raise
    store.record_send("stk_00000000")


# ------------------------------------------------------------------
# lookup_by_hash()
# ------------------------------------------------------------------


def test_lookup_by_hash_found(store: StickerStore) -> None:
    sid, _ = store.add(_JPEG_DATA, "desc", "hint")
    found = store.lookup_by_hash(_JPEG_DATA)
    assert found == sid


def test_lookup_by_hash_not_found(store: StickerStore) -> None:
    found = store.lookup_by_hash(_JPEG_DATA)
    assert found is None


def test_lookup_by_hash_different_content(store: StickerStore) -> None:
    store.add(_JPEG_DATA, "desc", "hint")
    found = store.lookup_by_hash(_JPEG_DATA_B)
    assert found is None


# ------------------------------------------------------------------
# format_prompt_view()
# ------------------------------------------------------------------


def test_format_prompt_view_empty(store: StickerStore) -> None:
    view = store.format_prompt_view()
    assert view == "当前表情包库为空"


def test_format_prompt_view_nonempty(store: StickerStore) -> None:
    sid, _ = store.add(_JPEG_DATA, "开心大笑", "搞笑对话时", source="auto")
    view = store.format_prompt_view()
    assert "当前表情包库：" in view
    assert f"[表情包:{sid}]" in view
    assert "开心大笑" in view
    assert "搞笑对话时" in view


def test_format_prompt_view_excludes_volatile_fields(store: StickerStore) -> None:
    """send_count, last_sent, created_at must NOT appear in the prompt view."""
    store.add(_JPEG_DATA, "desc", "hint")
    store.record_send(store.lookup_by_hash(_JPEG_DATA))  # type: ignore[arg-type]

    view = store.format_prompt_view()
    assert "send_count" not in view
    assert "last_sent" not in view
    assert "created_at" not in view


def test_format_prompt_view_multiple(store: StickerStore) -> None:
    store.add(_JPEG_DATA, "JPEG贴", "任何时候", source="auto")
    store.add(_PNG_DATA, "PNG贴", "悲伤时", source="auto")
    view = store.format_prompt_view()
    lines = view.splitlines()
    # header + 2 entries
    assert len(lines) == 3


# ------------------------------------------------------------------
# Index persistence across reload
# ------------------------------------------------------------------


def test_index_persists_across_reload(tmp_path) -> None:
    """Create store, add sticker, reload from same dir, verify data."""
    storage = str(tmp_path / "stickers")

    store1 = StickerStore(storage_dir=storage)
    sid, _ = store1.add(_JPEG_DATA, "持久化测试", "重启验证", source="admin")
    store1.record_send(sid)

    # Create new instance on same directory
    store2 = StickerStore(storage_dir=storage)

    entry = store2.get(sid)
    assert entry is not None
    assert entry["description"] == "持久化测试"
    assert entry["usage_hint"] == "重启验证"
    assert entry["source"] == "admin"
    assert entry["send_count"] == 1
    assert entry["last_sent"] is not None

    # File should still be accessible
    path = store2.resolve_path(sid)
    assert path is not None
    assert path.exists()
    assert path.read_bytes() == _JPEG_DATA


def test_index_persists_removal(tmp_path) -> None:
    storage = str(tmp_path / "stickers")

    store1 = StickerStore(storage_dir=storage)
    sid, _ = store1.add(_JPEG_DATA, "remove test", "hint")
    store1.remove(sid)

    store2 = StickerStore(storage_dir=storage)
    assert store2.get(sid) is None
    assert store2.list_all() == {}
