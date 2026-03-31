"""群聊上下文：记录群内最近消息，让 Bot 了解聊天语境。"""

from collections import defaultdict, deque
from typing import TypedDict


class GroupMessage(TypedDict):
    user_id: str
    nickname: str
    content: str


class GroupContext:
    def __init__(self, max_messages: int = 50) -> None:
        self._store: dict[str, deque[GroupMessage]] = defaultdict(lambda: deque(maxlen=max_messages))

    def add(self, group_id: str, user_id: str, nickname: str, content: str) -> None:
        self._store[group_id].append(GroupMessage(user_id=user_id, nickname=nickname, content=content))

    def get_recent(self, group_id: str, limit: int = 20) -> list[GroupMessage]:
        messages = list(self._store[group_id])
        return messages[-limit:]

    def format(self, group_id: str, limit: int = 20) -> str:
        messages = self.get_recent(group_id, limit)
        if not messages:
            return ""
        lines = [f"{m['nickname']}({m['user_id']}): {m['content']}" for m in messages]
        return "【最近群聊记录】\n" + "\n".join(lines)
