"""Group chat unified timeline: append-only turns + pending buffer model."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, overload

from src.memory.types import Content, ContentBlock, TextBlock

if TYPE_CHECKING:
    from src.memory.message_log import MessageLog

_MAX_GROUPS = 200


class TimelineMessage(TypedDict):
    role: Literal["user", "assistant"]
    speaker: str | None  # user → "昵称(QQ号)", assistant → None
    content: Content
    message_id: NotRequired[int | None]  # QQ message_id for reply references


# ------------------------------------------------------------------
# Module-level merge helpers (used by other modules)
# ------------------------------------------------------------------


def _merge_user_contents(batch: list[TimelineMessage]) -> Content:
    """Merge consecutive user messages into a single content value.

    Returns str if all messages are plain text (backward compatible).
    Returns list[ContentBlock] if any message contains image blocks.
    """
    has_blocks = any(isinstance(m["content"], list) for m in batch)

    if not has_blocks:
        lines: list[str] = []
        for m in batch:
            assert isinstance(m["content"], str)
            mid = m.get("message_id")
            tag = f"«msg:{mid}» " if mid is not None else ""
            if m["speaker"] is not None:
                lines.append(f"{tag}{m['speaker']}: {m['content']}")
            else:
                lines.append(f"{tag}{m['content']}" if tag else m["content"])
        return "\n".join(lines)

    merged: list[ContentBlock] = []
    for m in batch:
        mid = m.get("message_id")
        tag = f"«msg:{mid}» " if mid is not None else ""
        prefix = f"{tag}{m['speaker']}: " if m["speaker"] is not None else tag
        if isinstance(m["content"], str):
            merged.append(TextBlock(type="text", text=f"{prefix}{m['content']}"))
        else:
            # Insert speaker prefix: prepend to first text block, or add as own block
            if prefix and (not m["content"] or m["content"][0]["type"] != "text"):
                merged.append(TextBlock(type="text", text=prefix.rstrip()))
            for j, block in enumerate(m["content"]):
                if j == 0 and block["type"] == "text" and prefix:
                    merged.append(TextBlock(type="text", text=f"{prefix}{block['text']}"))
                else:
                    merged.append(block)
    return merged


def _merge_assistant_contents(parts: list[Content]) -> Content:
    """Merge consecutive assistant message contents into one.

    Returns str if all parts are plain text, otherwise list[ContentBlock].
    """
    if len(parts) == 1:
        return parts[0]

    has_blocks = any(isinstance(p, list) for p in parts)
    if not has_blocks:
        return "\n".join(p for p in parts if isinstance(p, str))

    merged: list[ContentBlock] = []
    for p in parts:
        if isinstance(p, str):
            merged.append(TextBlock(type="text", text=p))
        else:
            merged.extend(p)
    return merged


# ------------------------------------------------------------------
# _TurnLog — append-only Sequence[dict[str, Any]]
# ------------------------------------------------------------------


class _TurnLog(Sequence[dict[str, Any]]):
    """Append-only container for finalized Anthropic message turns."""

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: list[dict[str, Any]] = []

    # -- Sequence protocol --

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._data)

    # -- Mutators (append-only, no arbitrary mutation) --

    def append(self, turn: dict[str, Any]) -> None:
        self._data.append(turn)

    def compact_truncate(self, count: int) -> None:
        """Remove the first `count` turns."""
        self._data = self._data[count:]

    def reset(self) -> None:
        """Clear all turns."""
        self._data.clear()


# ------------------------------------------------------------------
# _GroupState
# ------------------------------------------------------------------


class _GroupState:
    __slots__ = (
        "last_cached_msg_index",
        "last_input_tokens",
        "pending",
        "summary",
        "turn_times",
        "turns",
    )

    def __init__(self) -> None:
        self.turns = _TurnLog()
        self.turn_times: list[float] = []
        self.pending: list[TimelineMessage] = []
        self.summary: str = ""
        self.last_input_tokens: int = 0
        self.last_cached_msg_index: int = 0


# ------------------------------------------------------------------
# GroupTimeline — public API
# ------------------------------------------------------------------


class GroupTimeline:
    """Group chat unified timeline with append-only turns and pending buffer."""

    def __init__(self, message_log: MessageLog | None = None) -> None:
        self._store: dict[str, _GroupState] = {}
        self._message_log = message_log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, group_id: str) -> _GroupState:
        if group_id not in self._store:
            if len(self._store) >= _MAX_GROUPS:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[group_id] = _GroupState()
        return self._store[group_id]

    # ------------------------------------------------------------------
    # Content extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_to_text(content: Content) -> str:
        """Extract plain text from Content for SQLite content_text column."""
        if isinstance(content, str):
            return content
        return " ".join(b["text"] for b in content if b["type"] == "text")

    @staticmethod
    def _content_to_json(content: Content) -> str:
        """Serialize Content to JSON for SQLite content_json column."""
        return json.dumps(content, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Message ingestion
    # ------------------------------------------------------------------

    def add(
        self,
        group_id: str,
        *,
        role: Literal["user", "assistant"],
        content: Content,
        speaker: str | None = None,
        message_id: int | None = None,
    ) -> None:
        """Add a message to the timeline.

        role="user"  → appends to pending buffer.
        role="assistant" → flushes pending into a merged user turn,
                           then appends both user and assistant turns to turns.
        """
        state = self._get_or_create(group_id)

        if role == "user":
            msg = TimelineMessage(role="user", speaker=speaker, content=content)
            if message_id is not None:
                msg["message_id"] = message_id
            state.pending.append(msg)
        else:
            # Flush pending user messages into a merged user turn
            now = time.time()
            if state.pending:
                user_content = _merge_user_contents(state.pending)
                state.turns.append({"role": "user", "content": user_content})
                state.turn_times.append(now)
                state.pending.clear()

            # Append assistant turn
            state.turns.append({"role": "assistant", "content": content})
            state.turn_times.append(now)

        # Fire-and-forget SQLite write
        if self._message_log:
            asyncio.create_task(self._message_log.record(  # noqa: RUF006
                group_id=group_id,
                role=role,
                speaker=speaker,
                content_text=self._content_to_text(content),
                content_json=self._content_to_json(content),
                message_id=message_id,
            ))

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    def get_turns(self, group_id: str) -> Sequence[dict[str, Any]]:
        """Return read-only view of finalized turns."""
        if group_id not in self._store:
            return _TurnLog()
        return self._store[group_id].turns

    def get_pending(self, group_id: str) -> list[TimelineMessage]:
        """Return a copy of the pending user message buffer."""
        if group_id not in self._store:
            return []
        return list(self._store[group_id].pending)

    def get_turn_time(self, group_id: str, index: int) -> float:
        """Return the timestamp of the turn at the given index."""
        if group_id not in self._store:
            return 0.0
        state = self._store[group_id]
        if 0 <= index < len(state.turn_times):
            return state.turn_times[index]
        return 0.0

    # ------------------------------------------------------------------
    # Backward-compatible accessors (used by client.py — will be removed)
    # ------------------------------------------------------------------

    def get_messages(self, group_id: str) -> list[TimelineMessage]:
        """Return raw messages list (DEPRECATED — use get_turns)."""
        if group_id not in self._store:
            return []
        state = self._store[group_id]
        # Reconstruct from turns + pending for backward compat
        result: list[TimelineMessage] = []
        for turn in state.turns:
            result.append(
                TimelineMessage(
                    role=turn["role"],
                    speaker=None,
                    content=turn["content"],
                )
            )
        result.extend(state.pending)
        return result

    def to_anthropic_messages(self, group_id: str) -> list[dict[str, Any]]:
        """Convert timeline to Anthropic messages format (DEPRECATED)."""
        if group_id not in self._store:
            return []
        state = self._store[group_id]

        # Build raw turn list (turns + any pending user messages)
        raw: list[dict[str, Any]] = list(state.turns)
        if state.pending:
            user_content = _merge_user_contents(state.pending)
            raw.append({"role": "user", "content": user_content})

        # Merge consecutive same-role turns (e.g. history reload may produce
        # consecutive assistant turns)
        if not raw:
            return []
        result: list[dict[str, Any]] = []
        i = 0
        while i < len(raw):
            role = raw[i]["role"]
            if role == "assistant":
                parts: list[Content] = []
                while i < len(raw) and raw[i]["role"] == "assistant":
                    parts.append(raw[i]["content"])
                    i += 1
                result.append({"role": "assistant", "content": _merge_assistant_contents(parts)})
            else:
                # User turns are already merged at flush time; just collect consecutive
                contents: list[Content] = []
                while i < len(raw) and raw[i]["role"] == "user":
                    contents.append(raw[i]["content"])
                    i += 1
                if len(contents) == 1:
                    result.append({"role": "user", "content": contents[0]})
                else:
                    # Merge multiple consecutive user turns (shouldn't normally happen
                    # but handle gracefully)
                    merged: list[ContentBlock] = []
                    for c in contents:
                        if isinstance(c, str):
                            merged.append(TextBlock(type="text", text=c))
                        else:
                            merged.extend(c)
                    if len(merged) == 1 and merged[0]["type"] == "text":
                        result.append({"role": "user", "content": merged[0].get("text", "")})
                    elif merged:
                        result.append({"role": "user", "content": merged})
                    else:
                        result.append({"role": "user", "content": ""})
        return result

    # ------------------------------------------------------------------
    # Summary & token management
    # ------------------------------------------------------------------

    def get_summary(self, group_id: str) -> str:
        if group_id not in self._store:
            return ""
        return self._store[group_id].summary

    def set_input_tokens(self, group_id: str, tokens: int) -> None:
        """Record input token count from the latest API call."""
        state = self._get_or_create(group_id)
        state.last_input_tokens = tokens

    def get_input_tokens(self, group_id: str) -> int:
        if group_id not in self._store:
            return 0
        return self._store[group_id].last_input_tokens

    def get_cached_msg_index(self, group_id: str) -> int:
        """Return the Anthropic messages index cached by the previous API call."""
        if group_id not in self._store:
            return 0
        return self._store[group_id].last_cached_msg_index

    def set_cached_msg_index(self, group_id: str, index: int) -> None:
        """Store which Anthropic messages index to use as cache breakpoint next call."""
        state = self._get_or_create(group_id)
        state.last_cached_msg_index = index

    def needs_compact(self, group_id: str, max_tokens: int, ratio: float) -> bool:
        """Check if compaction is needed: input tokens exceed threshold."""
        return self.get_input_tokens(group_id) > max_tokens * ratio

    # ------------------------------------------------------------------
    # Compaction & cleanup
    # ------------------------------------------------------------------

    def compact(self, group_id: str, split: int, new_summary: str) -> None:
        """Truncate the first `split` turns and update summary."""
        if group_id not in self._store:
            return
        state = self._store[group_id]
        state.turns.compact_truncate(split)
        state.turn_times = state.turn_times[split:]
        state.summary = new_summary
        state.last_input_tokens = 0
        state.last_cached_msg_index = 0

    def drop_oldest(self, group_id: str, count: int) -> None:
        """Drop the oldest `count` turns. For micro compact."""
        state = self._store.get(group_id)
        if state is None:
            return
        state.turns.compact_truncate(count)
        state.turn_times = state.turn_times[count:]

    def clear(self, group_id: str) -> None:
        """Clear turns + pending + turn_times, but preserve summary."""
        if group_id not in self._store:
            return
        state = self._store[group_id]
        state.turns.reset()
        state.turn_times.clear()
        state.pending.clear()
