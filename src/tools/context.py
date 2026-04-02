"""工具调用上下文：携带 Bot 实例和事件信息，供业务工具使用。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolContext:
    """每次对话调用时构建，传入工具执行。"""

    bot: Any = None  # nonebot.adapters.onebot.v11.Bot（避免顶层导入耦合）
    user_id: str = ""
    group_id: str | None = None
    session_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
