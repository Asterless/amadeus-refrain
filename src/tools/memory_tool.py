"""记忆工具：让 LLM 主动记住/回忆用户信息。"""

from typing import Any

from src.memory.long_term import LongTermMemory
from src.tools.base import Tool
from src.tools.context import ToolContext


class SaveMemoryTool(Tool):
    def __init__(self, long_term: LongTermMemory) -> None:
        self._long_term = long_term

    @property
    def name(self) -> str:
        return "save_memory"

    @property
    def description(self) -> str:
        return "保存关于用户的重要信息（昵称、特征、关键事件）到长期记忆，以便未来对话中使用。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户 QQ 号"},
                "nickname": {"type": "string", "description": "用户昵称或称呼"},
                "traits": {
                    "type": "object",
                    "description": "用户特征键值对，如 {'爱好': '编程', '性格': '开朗'}",
                    "additionalProperties": {"type": "string"},
                },
                "event": {"type": "string", "description": "值得记住的关键事件"},
            },
            "required": ["user_id"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        user_id: str = kwargs["user_id"]
        nickname: str | None = kwargs.get("nickname")
        traits: dict[str, str] | None = kwargs.get("traits")
        event: str | None = kwargs.get("event")

        if nickname or traits:
            await self._long_term.update_profile(user_id, nickname=nickname, traits=traits)
        if event:
            await self._long_term.add_event(user_id, event)

        parts: list[str] = []
        if nickname:
            parts.append(f"昵称 '{nickname}'")
        if traits:
            parts.append(f"特征 {traits}")
        if event:
            parts.append(f"事件 '{event}'")
        return f"已记住: {', '.join(parts)}" if parts else "没有需要保存的内容"


class RecallMemoryTool(Tool):
    def __init__(self, long_term: LongTermMemory) -> None:
        self._long_term = long_term

    @property
    def name(self) -> str:
        return "recall_memory"

    @property
    def description(self) -> str:
        return "回忆关于某个用户的长期记忆，包括画像和关键事件。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户 QQ 号"},
            },
            "required": ["user_id"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        user_id: str = kwargs["user_id"]
        content = await self._long_term.get_full_context(user_id)
        return content.strip() if content.strip() else f"没有关于用户 {user_id} 的记忆。"
