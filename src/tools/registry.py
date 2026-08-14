"""工具注册表：统一管理、查找、转换工具。"""

import json
from typing import Any

from loguru import logger

from src.tools.base import Tool
from src.tools.context import ToolContext


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai_tool() for t in self._tools.values()]

    async def call(self, name: str, arguments: str, ctx: ToolContext) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"未知工具: {name}"
        try:
            kwargs: dict[str, Any] = json.loads(arguments) if arguments else {}
            result = await tool.execute(ctx, **kwargs)
            if name == "send_sticker" and result.startswith("已发送"):
                ctx.extra["sticker_sent"] = True
            return result
        except Exception:
            logger.exception("tool error | name={}", name)
            return "工具执行出错，请稍后重试"

    @property
    def empty(self) -> bool:
        return len(self._tools) == 0

    @property
    def names(self) -> tuple[str, ...]:
        """Registered tool names in registration order, for diagnostics."""
        return tuple(self._tools)
