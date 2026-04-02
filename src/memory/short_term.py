"""短期记忆：每个会话累积对话历史，按需 compact。"""

from typing import Literal, TypedDict

_MAX_SESSIONS = 500


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class _SessionState:
    __slots__ = ("last_input_tokens", "messages", "summary")

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []
        self.summary: str = ""
        self.last_input_tokens: int = 0


class ShortTermMemory:
    def __init__(self) -> None:
        self._store: dict[str, _SessionState] = {}

    def _get_or_create(self, session_id: str) -> _SessionState:
        if session_id not in self._store:
            if len(self._store) >= _MAX_SESSIONS:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[session_id] = _SessionState()
        return self._store[session_id]

    def add(self, session_id: str, role: Literal["user", "assistant"], content: str) -> None:
        state = self._get_or_create(session_id)
        state.messages.append(ChatMessage(role=role, content=content))

    def get(self, session_id: str) -> list[ChatMessage]:
        if session_id not in self._store:
            return []
        return list(self._store[session_id].messages)

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def get_summary(self, session_id: str) -> str:
        if session_id not in self._store:
            return ""
        return self._store[session_id].summary

    def set_input_tokens(self, session_id: str, tokens: int) -> None:
        if session_id in self._store:
            self._store[session_id].last_input_tokens = tokens

    def get_input_tokens(self, session_id: str) -> int:
        if session_id not in self._store:
            return 0
        return self._store[session_id].last_input_tokens

    def needs_compact(self, session_id: str, max_tokens: int, ratio: float) -> bool:
        return self.get_input_tokens(session_id) > max_tokens * ratio

    def drop_oldest(self, session_id: str, count: int) -> None:
        """Drop the oldest `count` messages. For micro compact."""
        state = self._store.get(session_id)
        if state is None:
            return
        state.messages = state.messages[count:]

    def compact(self, session_id: str, split: int, new_summary: str) -> None:
        if session_id not in self._store:
            return
        state = self._store[session_id]
        state.messages = state.messages[split:]
        state.summary = new_summary
        state.last_input_tokens = 0
