"""短期记忆：每个会话保留最近 N 轮对话。"""

from collections import defaultdict, deque
from typing import Literal, TypedDict


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ShortTermMemory:
    def __init__(self, max_rounds: int = 20) -> None:
        self.max_rounds = max_rounds
        # key: session_id (group_id 或 "private_{user_id}")
        self._store: dict[str, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=max_rounds * 2)  # user+assistant 各一条算一轮
        )

    def add(self, session_id: str, role: Literal["user", "assistant"], content: str) -> None:
        self._store[session_id].append(ChatMessage(role=role, content=content))

    def get(self, session_id: str) -> list[ChatMessage]:
        return list(self._store[session_id])

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
