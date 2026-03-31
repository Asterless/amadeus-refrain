"""时间工具：查询当前日期时间。"""

from datetime import UTC, datetime
from typing import Any

from src.tools.base import Tool
from src.tools.context import ToolContext

_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


class DateTimeTool(Tool):
    @property
    def name(self) -> str:
        return "get_datetime"

    @property
    def description(self) -> str:
        return "获取当前的日期和时间。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        now = datetime.now(tz=UTC).astimezone()
        weekday = _WEEKDAYS[now.weekday()]
        return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {weekday}"
