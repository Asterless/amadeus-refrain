"""群管理工具：通过 OneBot API 执行群管理操作，需要 SUPERUSER 权限。"""

from typing import Any

from src.tools.base import Tool
from src.tools.context import ToolContext


def _check_auth(ctx: ToolContext, superusers: set[str]) -> str | None:
    """鉴权检查。返回 None 表示通过，否则返回错误信息。"""
    if not ctx.bot or not ctx.group_id:
        return "此操作仅在群聊中可用"
    if ctx.user_id not in superusers:
        return "权限不足: 仅管理员可执行此操作"
    return None


class MuteUserTool(Tool):
    def __init__(self, superusers: set[str]) -> None:
        self._superusers = superusers

    @property
    def name(self) -> str:
        return "mute_user"

    @property
    def description(self) -> str:
        return "在群里禁言指定用户。duration=0 表示解除禁言。需要机器人是管理员且请求者是 SUPERUSER。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "要禁言的用户 QQ 号"},
                "duration": {"type": "integer", "description": "禁言时长（秒），0=解除禁言，默认60", "default": 60},
            },
            "required": ["user_id"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        if err := _check_auth(ctx, self._superusers):
            return err
        assert ctx.group_id is not None
        user_id: str = kwargs["user_id"]
        duration: int = kwargs.get("duration", 60)
        await ctx.bot.set_group_ban(group_id=int(ctx.group_id), user_id=int(user_id), duration=duration)
        if duration == 0:
            return f"已解除 {user_id} 的禁言"
        return f"已禁言 {user_id} {duration}秒"


class SetTitleTool(Tool):
    def __init__(self, superusers: set[str]) -> None:
        self._superusers = superusers

    @property
    def name(self) -> str:
        return "set_title"

    @property
    def description(self) -> str:
        return "设置群成员的专属头衔。需要机器人是群主且请求者是 SUPERUSER。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "目标用户 QQ 号"},
                "title": {"type": "string", "description": "专属头衔内容，空字符串=清除头衔"},
            },
            "required": ["user_id", "title"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        if err := _check_auth(ctx, self._superusers):
            return err
        assert ctx.group_id is not None
        user_id: str = kwargs["user_id"]
        title: str = kwargs["title"]
        await ctx.bot.set_group_special_title(
            group_id=int(ctx.group_id), user_id=int(user_id), special_title=title
        )
        return f"已设置 {user_id} 的头衔为 '{title}'" if title else f"已清除 {user_id} 的头衔"


class SendGroupMsgTool(Tool):
    def __init__(self, superusers: set[str]) -> None:
        self._superusers = superusers

    @property
    def name(self) -> str:
        return "send_group_msg"

    @property
    def description(self) -> str:
        return "主动向指定群发送一条消息。用于通知、提醒等场景。需要请求者是 SUPERUSER。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "目标群号"},
                "message": {"type": "string", "description": "要发送的消息内容"},
            },
            "required": ["group_id", "message"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        if not ctx.bot:
            return "Bot 不可用"
        if ctx.user_id not in self._superusers:
            return "权限不足: 仅管理员可执行此操作"
        group_id: str = kwargs["group_id"]
        message: str = kwargs["message"]
        await ctx.bot.send_group_msg(group_id=int(group_id), message=message)
        return f"已发送消息到群 {group_id}"
