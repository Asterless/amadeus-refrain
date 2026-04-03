"""Sticker store: persistent storage and index for collected stickers."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Magic byte constants for image format detection
# ---------------------------------------------------------------------------

_MAGIC_JPEG = b"\xff\xd8\xff"
_MAGIC_PNG = b"\x89PNG"  # 89 50 4e 47
_MAGIC_WEBP_RIFF = b"RIFF"
_MAGIC_GIF87 = b"GIF87a"
_MAGIC_GIF89 = b"GIF89a"

_INDEX_FILE = "index.json"


def _detect_format(data: bytes) -> str:
    """Detect image format from magic bytes.

    Returns the file extension (without dot): 'jpg', 'png', 'webp'.
    Raises ValueError for GIF or unknown formats.
    """
    header = data[:6]

    # Reject GIF unconditionally
    if header[:6] in (_MAGIC_GIF87, _MAGIC_GIF89):
        raise ValueError("GIF format is not supported for stickers")

    if data[:3] == _MAGIC_JPEG:
        return "jpg"
    if data[:4] == _MAGIC_PNG:
        return "png"
    if data[:4] == _MAGIC_WEBP_RIFF:
        return "webp"

    raise ValueError(f"Unknown or unsupported image format (header: {data[:8].hex()})")


def _compute_hash(data: bytes) -> str:
    """Return the first 8 hex chars of SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()[:8]


class StickerStore:
    """Persistent sticker library backed by a directory and index.json."""

    def __init__(self, storage_dir: str, max_count: int = 200) -> None:
        self._storage_dir = Path(storage_dir)
        self._max_count = max_count
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Any] = self._load_index()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def storage_dir(self) -> Path:
        """Return the resolved storage directory path."""
        return self._storage_dir

    @property
    def max_count(self) -> int:
        """Return the configured maximum sticker count."""
        return self._max_count

    # ------------------------------------------------------------------
    # Index I/O
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self._storage_dir / _INDEX_FILE

    def _load_index(self) -> dict[str, Any]:
        """Load index from disk. Returns empty index if file doesn't exist."""
        path = self._index_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return dict(data.get("stickers", {}))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_index(self) -> None:
        """Persist the current index to disk atomically."""
        path = self._index_path()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"stickers": self._index}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_all(self) -> dict[str, Any]:
        """Return a copy of all sticker entries."""
        return dict(self._index)

    def get(self, sticker_id: str) -> dict[str, Any] | None:
        """Return entry for the given sticker_id, or None if not found."""
        return self._index.get(sticker_id)

    def resolve_path(self, sticker_id: str) -> Path | None:
        """Return the absolute file path for the sticker, or None if not found."""
        entry = self._index.get(sticker_id)
        if entry is None:
            return None
        return self._storage_dir / entry["file"]

    def lookup_by_hash(self, image_data: bytes) -> str | None:
        """Return sticker_id if image_data matches an existing sticker, else None."""
        sticker_id = "stk_" + _compute_hash(image_data)
        if sticker_id in self._index:
            return sticker_id
        return None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(
        self,
        image_data: bytes,
        description: str,
        usage_hint: str,
        source: str = "auto",
    ) -> tuple[str, bool]:
        """Add a sticker to the store.

        Rejects GIFs and unknown formats.
        Deduplicates by content hash.

        Returns:
            (sticker_id, is_new) — is_new=False means it already existed.
        """
        ext = _detect_format(image_data)  # raises ValueError for GIF / unknown
        hash_prefix = _compute_hash(image_data)
        sticker_id = f"stk_{hash_prefix}"

        # Dedup: already known
        if sticker_id in self._index:
            return (sticker_id, False)

        filename = f"{sticker_id}.{ext}"
        file_path = self._storage_dir / filename
        file_path.write_bytes(image_data)

        now_iso = datetime.now(UTC).isoformat()
        self._index[sticker_id] = {
            "file": filename,
            "description": description,
            "usage_hint": usage_hint,
            "source": source,
            "send_count": 0,
            "last_sent": None,
            "created_at": now_iso,
        }
        self._save_index()
        return (sticker_id, True)

    def remove(self, sticker_id: str) -> bool:
        """Remove sticker from index and delete its file.

        Returns True if the sticker was found and removed, False otherwise.
        """
        entry = self._index.pop(sticker_id, None)
        if entry is None:
            return False

        file_path = self._storage_dir / entry["file"]
        if file_path.exists():
            file_path.unlink()

        self._save_index()
        return True

    def record_send(self, sticker_id: str) -> None:
        """Increment send_count and update last_sent timestamp."""
        entry = self._index.get(sticker_id)
        if entry is None:
            return
        entry["send_count"] = entry.get("send_count", 0) + 1
        entry["last_sent"] = datetime.now(UTC).isoformat()
        self._save_index()

    # ------------------------------------------------------------------
    # Prompt injection
    # ------------------------------------------------------------------

    def format_prompt_view(self) -> str:
        """Return a compact view of the sticker library for system prompt injection.

        Only includes id, description, and usage_hint — volatile fields
        (send_count, last_sent, created_at) are excluded for prompt cache stability.
        """
        if not self._index:
            return "当前表情包库为空"

        lines = ["当前表情包库："]
        for sticker_id, entry in self._index.items():
            description = entry.get("description", "")
            usage_hint = entry.get("usage_hint", "")
            lines.append(f"[表情包:{sticker_id}] {description} | {usage_hint}")

        return "\n".join(lines)
