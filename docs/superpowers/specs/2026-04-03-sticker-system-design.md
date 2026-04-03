# Sticker System — Design Spec

Date: 2026-04-03

## Goal

Give the bot a **self-evolving sticker library**: it can collect stickers from
group chat, receive admin-fed stickers, autonomously decide when to send one,
and periodically clean up low-frequency stickers via the existing Dream agent.

## Scope

- **In scope:** sticker storage & index, `save_sticker` / `send_sticker` tools,
  Dream-based LRU cleanup, index injection into system prompt, GIF exclusion,
  behavioral guardrails in soul instructions.
- **Out of scope:** AI image generation, web image search, GIF/animated sticker
  support, inline text+image mixed messages (sticker sent as separate message).

---

## Storage

### File Layout

```
storage/stickers/
├── index.json          # catalog
├── stk_a1b2c3d4.jpg   # flat storage, content-hash filename
├── stk_e5f6g7h8.jpg
└── ...
```

- Files are **always** renamed to `stk_{short_hash}.{ext}` regardless of the
  original filename. The hash is derived from image content (e.g. first 8 hex
  chars of SHA-256), providing natural deduplication.
- Only static image formats (JPEG, PNG, WebP). **GIF is excluded** — vision can
  only see the first frame, making understanding unreliable.

### index.json Schema

```json
{
  "stickers": {
    "stk_a1b2c3d4": {
      "file": "stk_a1b2c3d4.jpg",
      "description": "一只猫咪露出嫌弃的表情",
      "usage_hint": "当对方说了什么无语的话时使用",
      "source": "auto | admin",
      "send_count": 12,
      "last_sent": "2026-04-01T14:30:00",
      "created_at": "2026-03-15T10:00:00"
    }
  }
}
```

| Field | Purpose |
|-------|---------|
| `file` | Filename relative to `storage/stickers/` |
| `description` | LLM-generated image description (via vision at save time) |
| `usage_hint` | LLM-written guidance on when to use this sticker |
| `source` | `"auto"` (bot self-collected) or `"admin"` (admin-fed) |
| `send_count` | Total times sent, used for LRU |
| `last_sent` | ISO timestamp of last send, used for LRU |
| `created_at` | ISO timestamp of collection |

---

## Conversation Tools

Two tools available during chat. No delete tool in conversation — cleanup is
delegated to Dream.

### `save_sticker`

**Purpose:** Collect a sticker from the current conversation.

**Input:**
- `image_ref` (string) — image reference ID from the current conversation
  (matches an `ImageRefBlock` in the message history)
- `description` (string) — what the image depicts
- `usage_hint` (string) — when this sticker should be used

**Execution:**
1. Resolve `image_ref` to disk path via `ImageCache`.
2. Check file header magic bytes — reject if GIF (`GIF87a` / `GIF89a`).
3. Compute content hash (SHA-256, first 8 hex chars) → `stk_{hash}`.
4. If hash already exists in index → return "already collected" (dedup).
5. Copy image to `storage/stickers/stk_{hash}.{ext}`.
6. Add entry to `index.json`.
7. Determine `source`: `"admin"` if caller is admin, else `"auto"`.

**Behavioral guardrail (enforced via soul instructions, not code):**
The bot must only call `save_sticker` when it:
- Fully understands the sticker's meaning and emotional context
- Can articulate a clear usage scenario
- Believes the sticker fits its own personality and will actually want to
  send it in the future

### `send_sticker`

**Purpose:** Send a sticker as a separate image message.

**Input:**
- `sticker_id` (string) — ID from the index (e.g. `stk_a1b2c3d4`)

**Execution:**
1. Validate `sticker_id` exists in index.
2. Resolve full file path.
3. Send image message via bot API (`bot.send()` or
   `bot.send_group_msg()` with image segment) — **side-effect, separate
   message from text**.
4. Update `send_count` += 1 and `last_sent` in index.
5. Return success/failure confirmation to LLM.

**Sending timing:** Fully autonomous. The bot decides when a sticker fits
the conversation flow. No user trigger required.

---

## System Prompt Injection

### Cache Impact Analysis

Current cache layout (4 breakpoints, all used):

```
① tools[-1]                    — global shared
② system block 1: personality  — global shared, built at startup
③ system block 2: index+memo   — per-entity, built per chat()
④ messages[near-end]           — per-conversation
```

The sticker index is injected into **block ③** (entity block, inside
`PromptBuilder.build_blocks()`). This is the only viable position — block ②
is a startup-time static block, and all 4 breakpoints are already occupied.

**Critical:** The injected view includes ONLY stable fields (`id`,
`description`, `usage_hint`). Volatile fields (`send_count`, `last_sent`,
`created_at`) are **excluded** from the prompt. This ensures:

- `send_sticker` calls do NOT change the prompt → block ③ stays cached
- `save_sticker` (rare, adds a new entry) → block ③ misses once, re-caches
- Dream cleanup (periodic, off-conversation) → no impact during chat
- Block ② is completely unaffected

### Index View

Injected at the end of entity block text in `build_blocks()`. Compact
one-line-per-sticker format to minimize token usage:

```
当前表情包库：
<stk:a1b2c3d4> 一只猫咪露出嫌弃的表情 | 对方说了无语的话时
<stk:e5f6g7h8> 熊猫头竖中指 | 阴阳怪气回怼时
```

Format: `<stk:hash> description | usage_hint`

When the library is empty: inject `当前表情包库为空`.

### Soul Instructions

Add to `soul/instruction.md`:
- Collect stickers only when you genuinely understand and connect with them
- Do not hoard — only save stickers you foresee yourself wanting to send
- Send stickers naturally as part of conversation rhythm; do not overuse
- Stickers are sent as separate messages (not inline with text)

---

## Dream Integration

Extend the existing `DreamAgent` to manage sticker library hygiene.

### Additional Dream Tools

- `list_stickers` — returns the full `index.json` content
- `delete_sticker(id)` — removes file from disk + entry from index

### Dream Prompt Extension

Add sticker maintenance instructions to Dream's system prompt:
- Review stickers with low `send_count` and old `created_at` (LRU candidates)
- Use judgment — a rarely-sent but unique/valuable sticker may be worth keeping
- Verify `description` and `usage_hint` accuracy; update if needed
- Keep the library lean and high-quality

No hard LRU threshold — the Dream LLM decides what to cull based on library
size, usage patterns, and sticker quality.

---

## GIF Exclusion

GIF detection happens at `save_sticker` time:
- Read first 6 bytes of the source image file.
- If bytes match `GIF87a` or `GIF89a` → reject with message explaining GIFs
  are not supported.
- This applies to both auto-collection and admin-fed stickers.

---

## Configuration

Add to `BotConfig` (in `config.py`):

```toml
[sticker]
enabled = true
storage_dir = "storage/stickers"
max_count = 200            # max stickers in library; Dream enforces
```

Minimal config — most behavior is LLM-driven via soul instructions, not
hard-coded thresholds.
