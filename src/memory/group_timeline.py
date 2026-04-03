"""群聊统一时间线：合并 GroupContext 与群组的 ShortTermMemory。"""

from typing import Any, Literal, TypedDict

from src.memory.types import Content, ContentBlock, TextBlock

_MAX_GROUPS = 200


class TimelineMessage(TypedDict):
    role: Literal["user", "assistant"]
    speaker: str | None  # user → "昵称(QQ号)", assistant → None
    content: Content


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
            if m["speaker"] is not None:
                lines.append(f"{m['speaker']}: {m['content']}")
            else:
                lines.append(m["content"])
        return "\n".join(lines)

    merged: list[ContentBlock] = []
    for m in batch:
        prefix = f"{m['speaker']}: " if m["speaker"] is not None else ""
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


class _GroupState:
    __slots__ = ("last_cached_msg_index", "last_input_tokens", "messages", "summary")

    def __init__(self) -> None:
        self.messages: list[TimelineMessage] = []
        self.summary: str = ""
        self.last_input_tokens: int = 0
        self.last_cached_msg_index: int = 0


class GroupTimeline:
    """群聊统一时间线，兼具上下文记录与 compact 能力。"""

    def __init__(self) -> None:
        self._store: dict[str, _GroupState] = {}

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_or_create(self, group_id: str) -> _GroupState:
        if group_id not in self._store:
            if len(self._store) >= _MAX_GROUPS:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[group_id] = _GroupState()
        return self._store[group_id]

    # ------------------------------------------------------------------
    # 消息管理
    # ------------------------------------------------------------------

    def add(
        self,
        group_id: str,
        *,
        role: Literal["user", "assistant"],
        content: Content,
        speaker: str | None = None,
    ) -> None:
        """追加一条消息；由 compact 控制大小，不做硬截断。"""
        state = self._get_or_create(group_id)
        state.messages.append(TimelineMessage(role=role, speaker=speaker, content=content))

    def get_messages(self, group_id: str) -> list[TimelineMessage]:
        """返回原始消息列表的副本。"""
        if group_id not in self._store:
            return []
        return list(self._store[group_id].messages)

    # ------------------------------------------------------------------
    # Anthropic 消息格式转换
    # ------------------------------------------------------------------

    def to_anthropic_messages(self, group_id: str) -> list[dict[str, Any]]:
        """将时间线转为 Anthropic messages 格式。"""
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
                user_batch: list[TimelineMessage] = []
                while i < len(messages) and messages[i]["role"] == "user":
                    user_batch.append(messages[i])
                    i += 1
                result.append({"role": "user", "content": _merge_user_contents(user_batch)})

        return result

    # ------------------------------------------------------------------
    # 摘要 & Token 管理
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
        """判断是否需要 compact：当前 input tokens 超过阈值。"""
        return self.get_input_tokens(group_id) > max_tokens * ratio

    def compact(self, group_id: str, split: int, new_summary: str) -> None:
        """裁剪前 split 条消息，更新摘要，重置 token 计数。"""
        if group_id not in self._store:
            return
        state = self._store[group_id]
        state.messages = state.messages[split:]
        state.summary = new_summary
        state.last_input_tokens = 0
        state.last_cached_msg_index = 0

    def drop_oldest(self, group_id: str, count: int) -> None:
        """Drop the oldest `count` messages. For micro compact."""
        state = self._store.get(group_id)
        if state is None:
            return
        state.messages = state.messages[count:]
