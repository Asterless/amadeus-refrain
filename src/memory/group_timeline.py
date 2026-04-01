"""群聊统一时间线：合并 GroupContext 与群组的 ShortTermMemory。"""

import time
from typing import Literal, TypedDict

_MAX_GROUPS = 200


class TimelineMessage(TypedDict):
    role: Literal["user", "assistant"]
    speaker: str | None  # user → "昵称(QQ号)", assistant → None
    content: str


class _GroupState:
    __slots__ = ("_max", "last_api_call_time", "last_input_tokens", "messages", "new_msg_count", "summary")

    def __init__(self, max_messages: int) -> None:
        self._max = max_messages
        self.messages: list[TimelineMessage] = []
        self.summary: str = ""
        self.last_input_tokens: int = 0
        self.last_api_call_time: float = 0.0
        self.new_msg_count: int = 0


class GroupTimeline:
    """群聊统一时间线，兼具上下文记录与 compact 能力。"""

    def __init__(self, max_messages: int = 50) -> None:
        self._max = max_messages
        self._store: dict[str, _GroupState] = {}

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_or_create(self, group_id: str) -> _GroupState:
        if group_id not in self._store:
            if len(self._store) >= _MAX_GROUPS:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[group_id] = _GroupState(self._max)
        return self._store[group_id]

    # ------------------------------------------------------------------
    # 消息管理
    # ------------------------------------------------------------------

    def add(
        self,
        group_id: str,
        *,
        role: Literal["user", "assistant"],
        content: str,
        speaker: str | None = None,
    ) -> None:
        """追加一条消息；超出上限时淘汰最旧的消息。"""
        state = self._get_or_create(group_id)
        state.messages.append(TimelineMessage(role=role, speaker=speaker, content=content))
        if len(state.messages) > state._max:
            state.messages = state.messages[-state._max :]
        if role == "user":
            state.new_msg_count += 1

    def get_messages(self, group_id: str) -> list[TimelineMessage]:
        """返回原始消息列表的副本。"""
        if group_id not in self._store:
            return []
        return list(self._store[group_id].messages)

    # ------------------------------------------------------------------
    # Anthropic 消息格式转换
    # ------------------------------------------------------------------

    def to_anthropic_messages(self, group_id: str) -> list[dict[str, str]]:
        """将时间线转为 Anthropic messages 格式。

        连续的 user 消息合并为一个 block，格式为 "speaker: content\\n"（末尾不加换行）。
        assistant 消息保持独立。
        """
        messages = self.get_messages(group_id)
        if not messages:
            return []

        result: list[dict[str, str]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "assistant":
                result.append({"role": "assistant", "content": msg["content"]})
                i += 1
            else:
                # 收集连续的 user 消息
                lines: list[str] = []
                while i < len(messages) and messages[i]["role"] == "user":
                    m = messages[i]
                    if m["speaker"] is not None:
                        lines.append(f"{m['speaker']}: {m['content']}")
                    else:
                        lines.append(m["content"])
                    i += 1
                result.append({"role": "user", "content": "\n".join(lines)})

        return result

    # ------------------------------------------------------------------
    # 摘要 & Token 管理
    # ------------------------------------------------------------------

    def get_summary(self, group_id: str) -> str:
        if group_id not in self._store:
            return ""
        return self._store[group_id].summary

    def set_input_tokens(self, group_id: str, tokens: int) -> None:
        """记录本次 API 调用的输入 token 数，同时刷新计时器并清零新消息计数。"""
        state = self._get_or_create(group_id)
        state.last_input_tokens = tokens
        state.last_api_call_time = time.monotonic()
        state.new_msg_count = 0

    def get_input_tokens(self, group_id: str) -> int:
        if group_id not in self._store:
            return 0
        return self._store[group_id].last_input_tokens

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

    # ------------------------------------------------------------------
    # 缓存预热支持
    # ------------------------------------------------------------------

    def should_warm(self, group_id: str, interval: int, ttl: float) -> bool:
        """判断是否应触发缓存预热。

        条件：
        1. 自上次 API 调用后新消息数 >= interval
        2. 曾有过 API 调用（last_api_call_time > 0）
        3. 上次 API 调用距今 < ttl 秒（会话仍活跃）
        """
        if group_id not in self._store:
            return False
        state = self._store[group_id]
        if state.last_api_call_time <= 0:
            return False
        if state.new_msg_count < interval:
            return False
        elapsed = time.monotonic() - state.last_api_call_time
        return elapsed < ttl

    def reset_warm_counter(self, group_id: str) -> None:
        """重置新消息计数器（预热触发后调用）。"""
        if group_id in self._store:
            self._store[group_id].new_msg_count = 0
