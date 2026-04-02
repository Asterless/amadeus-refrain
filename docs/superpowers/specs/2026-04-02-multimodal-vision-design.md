# Multimodal Vision Support — Design Spec

Date: 2026-04-02

## Goal

Allow the bot to **understand** images and QQ stickers (表情包) sent by users in
both private and group chats. The bot receives image content, passes it to
Claude Vision, and responds with text. Image *sending* is out of scope for this
iteration but the architecture should not preclude it.

## Scope

- **In scope:** receiving images, QQ face emoji mapping, Claude Vision
  integration, disk-based image cache, configuration, graceful degradation.
- **Out of scope:** bot sending images/stickers, video/audio, file attachments.

---

## Architecture Overview

```
QQ message (OneBot V11 segments)
  │
  ├─ seg.type == "text"   → append to text
  ├─ seg.type == "at"     → append @QQ to text
  ├─ seg.type == "face"   → lookup QQ_FACE dict → append [表情名] to text
  ├─ seg.type == "image"  → ImageCache.save(url, file_id) → ImageRef
  └─ other                → ignore or append [不支持的消息类型]
  │
  ▼
MessageContent { text: str, images: list[ImageRef] }
  │
  ▼
Store in history (short_term / group_timeline)
  content: str | list[ContentBlock]
  ContentBlock = TextBlock | ImageRefBlock
  ImageRefBlock stores disk path only, NOT base64
  │
  ▼
Build Anthropic API request (_build_*_messages)
  image_ref → ImageCache.load_as_base64() → Anthropic image block
  missing file → degrade to {"type": "text", "text": "[图片已过期]"}
```

---

## 1. Message Parsing Layer

**File:** `src/plugins/chat/__init__.py` — `_render_message()`

Current signature returns `str`. Change to return a structured result:

```python
@dataclass
class ImageRef:
    path: Path          # e.g. storage/image_cache/ab/ab3f7c...image
    media_type: str     # e.g. "image/jpeg"

@dataclass
class MessageContent:
    text: str
    images: list[ImageRef]
```

Segment handling:

| Segment type | Action |
|---|---|
| `text` | Append `seg.data["text"]` to text |
| `at` | Append `@{qq}` to text |
| `face` | Lookup `QQ_FACE[id]` → append `[名称]` to text; fallback `[表情]` |
| `image` | Download via `ImageCache.save()` → append to images; on failure append `[图片]` to text |
| other | Ignore or append `[不支持的消息类型]` to text |

**Per-message image limit:** max `max_images_per_message` images per single
message. Excess images become `[图片]` text.

When `vision.enabled = false`, skip all image downloads; every image segment
becomes `[图片]` text. Face mapping is always active (zero-cost text transform).

---

## 2. Image Cache Module

**New file:** `src/memory/image_cache.py`

```
ImageCache
  __init__(cache_dir: Path)

  async save(url: str, file_id: str) -> ImageRef | None
    # Download → resize with pyvips → save to disk
    # Returns None on failure (network, decode, etc.)

  load_as_base64(ref: ImageRef) -> dict | None
    # Read from disk → base64 encode → Anthropic image content block
    # Returns None if file missing

  cleanup(max_age: timedelta)
    # Delete files older than max_age
```

### Storage layout

Two-level hash directory using the first two characters of `file_id`:

```
storage/image_cache/
  ab/
    ab3f7c...image
  1e/
    1e92d4...image
```

256 buckets, prevents single-directory I/O degradation.

### Cache key

Use OneBot V11's `seg.data["file"]` field as the cache key. This is a
QQ-assigned unique identifier for each image, stable across reconnections.
Benefits:

- Bot online → download, cache with file ID → store reference
- Bot reconnects, pulls history → same file ID → cache hit, no re-download
- Cache miss on reconnect → attempt URL download → fail gracefully to `[图片已过期]`

### Image processing

- **Library:** pyvips (libvips) — stream-based, 5–10x faster than Pillow, low
  memory footprint.
- **Resize:** Scale so longest edge ≤ `max_dimension` (default 768px).
- **Output format:** JPEG (smaller size). GIF/animated images → extract first
  frame → JPEG (Claude Vision does not support animation).
- **System dependency:** `libvips` (one `apt-get install` line in Dockerfile).
- **Python dependency:** `pyvips` added to `pyproject.toml`.

### Cleanup

- Run once on bot startup.
- Periodic cleanup during runtime (simple background task or opportunistic on
  each `save()` call).
- Default max age: 24 hours (configurable).

---

## 3. History Storage Layer

**Files:** `src/memory/short_term.py`, `src/memory/group_timeline.py`

### Type changes

```python
class TextBlock(TypedDict):
    type: Literal["text"]
    text: str

class ImageRefBlock(TypedDict):
    type: Literal["image_ref"]
    path: str           # disk path relative to project root
    media_type: str

ContentBlock = TextBlock | ImageRefBlock

class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]    # was: str

class TimelineMessage(TypedDict):
    role: Literal["user", "assistant"]
    speaker: str | None
    content: str | list[ContentBlock]    # was: str
```

### Group timeline merge logic

`to_anthropic_messages()` merges consecutive user messages. With multimodal
content, merge by concatenating content block arrays. Each message's text block
gets prefixed with `speaker: `.

### History loader

`history_loader.py` — when loading historical messages via NapCat API:

- Extract image segments, attempt download via `ImageCache.save()`
- Cache hit (by file ID) → use cached file
- Cache miss + URL still valid → download and cache
- Cache miss + URL expired → degrade to `[图片已过期]`

---

## 4. API Request Assembly

**File:** `src/llm/client.py`

### `_to_anthropic_message()` changes

When `content` is a `list[ContentBlock]`:

- `TextBlock` → pass through as `{"type": "text", "text": "..."}`
- `ImageRefBlock` → call `ImageCache.load_as_base64(ref)`
  - Success → `{"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}`
  - Failure → `{"type": "text", "text": "[图片已过期]"}`

When `content` is `str` → existing behavior, backward compatible.

### Per-request image cap

After converting all messages, count total image blocks across the entire
request. If exceeding `max_images_per_request`, replace oldest images (from
earliest messages) with `[图片]` text blocks.

### Cache control

Existing `cache_control` placement logic unchanged. When the second-to-last
message contains image blocks, apply `cache_control` to the last block in its
content array (Anthropic supports `cache_control` on both text and image
blocks). Cache hits are not affected because:

- Same disk file → same base64 → identical content prefix
- File ID keying prevents re-downloads that could produce different bytes

---

## 5. QQ Face Mapping

**New file:** `src/constants/qq_face.py`

Static dict mapping ~200+ QQ classic face IDs to Chinese names:

```python
QQ_FACE: dict[int, str] = {
    0: "惊讶",
    1: "撇嘴",
    2: "色",
    ...
    178: "捂脸",
    ...
}
```

Usage: `QQ_FACE.get(int(face_id), "表情")` → `[捂脸]`

IDs not in the dict fall back to `[表情]`.

---

## 6. Configuration

**Location:** `config.toml` `[vision]` section, mapped to `BotConfig`.

```toml
[vision]
enabled = true
max_images_per_message = 5
max_images_per_request = 15
max_dimension = 768
cache_dir = "storage/image_cache"
cache_max_age_hours = 24
```

`enabled = false` short-circuits the entire pipeline: message parsing skips
image downloads, all image segments become `[图片]` text. Face mapping remains
active regardless (zero-cost text operation).

---

## 7. Error Handling & Edge Cases

**Guiding principle:** Images enhance but are never required. Any failure
degrades silently to a text placeholder. Core conversation is never blocked.

| Scenario | Behavior |
|---|---|
| Image download fails (timeout, invalid URL) | Degrade to `[图片]`, log warning |
| Unsupported image format (decode failure) | Degrade to `[图片]`, log warning |
| GIF / animated image | Extract first frame → static JPEG |
| Bot reconnect, pull history, cache hit | Reuse cached file |
| Bot reconnect, cache miss, URL valid | Download and cache |
| Bot reconnect, cache miss, URL expired | Degrade to `[图片已过期]` |
| Cache cleanup race (file deleted during read) | `load_as_base64()` returns None → `[图片已过期]` |
| Vision disabled in config | All images → `[图片]` text, no downloads |

---

## New Dependencies

| Package | Purpose | Type |
|---|---|---|
| `pyvips` | Fast image resize via libvips | Python (pyproject.toml) |
| `libvips` | Image processing C library | System (Dockerfile apt-get) |

---

## Files Changed / Created

| File | Change |
|---|---|
| `src/plugins/chat/__init__.py` | `_render_message()` returns `MessageContent`; handle `image`, `face` segments |
| `src/memory/image_cache.py` | **New** — download, resize, cache, load, cleanup |
| `src/memory/short_term.py` | `ChatMessage.content` type union; accept content blocks |
| `src/memory/group_timeline.py` | `TimelineMessage.content` type union; merge logic for blocks |
| `src/memory/history_loader.py` | Extract image segments from historical messages |
| `src/llm/client.py` | `_to_anthropic_message()` handles content blocks; per-request image cap |
| `src/constants/qq_face.py` | **New** — QQ face ID → name mapping dict |
| `src/config_loader.py` | Add `VisionConfig` to `BotConfig` |
| `pyproject.toml` | Add `pyvips` dependency |
| `Dockerfile` | Add `apt-get install libvips` |
| `config.example.toml` | Add `[vision]` section |
