"""Sticker tools: save, send, and manage stickers in the library."""

from pathlib import Path
from typing import Any

from loguru import logger

from src.sticker.store import StickerStore
from src.tools.base import Tool
from src.tools.context import ToolContext


class SaveStickerTool(Tool):
    """Save an image from the conversation to the sticker library."""

    def __init__(self, store: StickerStore, superusers: set[str]) -> None:
        self._store = store
        self._superusers = superusers

    @property
    def name(self) -> str:
        return "save_sticker"

    @property
    def description(self) -> str:
        return (
            "收录一张对话中的图片到你的表情包库。"
            "image_tag 使用图片旁边的 «img:N» 标签。"
            "只在管理员要求时才调用，必须将管理员QQ号填入 requested_by。"
            "只在你完全理解图片含义、清楚使用场景、且符合自己性格时才调用。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_tag": {
                    "type": "string",
                    "description": "对话中图片的标签，如 img:3",
                },
                "description": {
                    "type": "string",
                    "description": "表情包内容描述",
                },
                "usage_hint": {
                    "type": "string",
                    "description": "适合使用该表情包的场景说明",
                },
                "requested_by": {
                    "type": "string",
                    "description": "发起请求的用户QQ号（群聊中从消息上下文提取）",
                },
            },
            "required": ["image_tag", "description", "usage_hint", "requested_by"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        image_tag: str = kwargs["image_tag"]
        description: str = kwargs["description"]
        usage_hint: str = kwargs["usage_hint"]

        requester = ctx.user_id or kwargs.get("requested_by", "")
        if requester not in self._superusers:
            return "只有管理员可以收录表情包"

        tag_map: dict[str, str] = ctx.extra.get("image_tags", {})
        path_str = tag_map.get(image_tag)
        if not path_str:
            return f"图片标签不存在: {image_tag}（可用标签: {', '.join(tag_map) or '无'}）"

        path = Path(path_str)
        if not path.exists():
            return f"图片文件已过期: {image_tag}"

        image_data = path.read_bytes()

        try:
            stk_id, is_new = self._store.add(image_data, description, usage_hint, source="admin")
        except ValueError as e:
            return f"无法收录: {e}"

        if not is_new:
            return f"表情包已存在: {stk_id}"

        # Notify timeline so the model can use this sticker in future calls
        # (system blocks are read-only, rebuilt only on compact)
        timeline = ctx.extra.get("timeline")
        if timeline and ctx.group_id:
            timeline.add(
                ctx.group_id,
                role="user",
                speaker="【系统】",
                content=f"新增表情包 «表情包:{stk_id}» {description} | {usage_hint}",
            )

        return f"{stk_id} 已收录"


class ManageStickerTool(Tool):
    """Update or delete stickers in the library."""

    def __init__(self, store: StickerStore, superusers: set[str]) -> None:
        self._store = store
        self._superusers = superusers

    @property
    def name(self) -> str:
        return "manage_sticker"

    @property
    def description(self) -> str:
        return (
            "管理表情包库：更新表情包的描述/场景说明，或删除表情包。"
            "必须从对话上下文识别是谁在要求操作，将其QQ号填入 requested_by。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sticker_id": {
                    "type": "string",
                    "description": "���情包 ID，如 stk_a1b2c3d4",
                },
                "action": {
                    "type": "string",
                    "enum": ["update", "delete"],
                    "description": "操作类型",
                },
                "requested_by": {
                    "type": "string",
                    "description": "发起请求的用户QQ号（群聊中从消息上下文提取）",
                },
                "description": {
                    "type": "string",
                    "description": "新的内容描述（仅 update 时使用）",
                },
                "usage_hint": {
                    "type": "string",
                    "description": "新的场景说明（仅 update 时使��）",
                },
            },
            "required": ["sticker_id", "action", "requested_by"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        sticker_id: str = kwargs["sticker_id"]
        action: str = kwargs["action"]

        if action == "delete":
            requester = ctx.user_id or kwargs.get("requested_by", "")
            if requester not in self._superusers:
                return "只有管理员可以删除表情包"
            if self._store.remove(sticker_id):
                return f"{sticker_id} 已删除"
            return f"表情包不存在: {sticker_id}"

        if action == "update":
            description: str | None = kwargs.get("description")
            usage_hint: str | None = kwargs.get("usage_hint")
            if description is None and usage_hint is None:
                return "请提供 description 或 usage_hint"
            if self._store.update(sticker_id, description, usage_hint):
                return f"{sticker_id} 已更新"
            return f"表情包不存在: {sticker_id}"

        return f"未知操作: {action}"


class SendStickerTool(Tool):
    """Send a sticker from the library as a standalone image message."""

    def __init__(self, store: StickerStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "send_sticker"

    @property
    def description(self) -> str:
        return "发送一张表情包（作为单独的图片消息）。从表情包库中选择合适的表情包发送。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sticker_id": {
                    "type": "string",
                    "description": "表情包 ID，如 stk_a1b2c3d4",
                },
            },
            "required": ["sticker_id"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        sticker_id: str = kwargs["sticker_id"]

        if not ctx.bot:
            return "Bot 不可用"

        file_path = self._store.resolve_path(sticker_id)
        if file_path is None:
            return f"表情包不存在: {sticker_id}"

        from nonebot.adapters.onebot.v11 import MessageSegment

        img_seg = MessageSegment.image(file_path)
        img_seg.data["subType"] = 1

        try:
            if ctx.group_id:
                await ctx.bot.send_group_msg(group_id=int(ctx.group_id), message=img_seg)
            else:
                await ctx.bot.send_private_msg(user_id=int(ctx.user_id), message=img_seg)
        except Exception as e:
            logger.error("send_sticker failed for {}: {}", sticker_id, e)
            return f"发送失败: {sticker_id}"

        self._store.record_send(sticker_id)
        logger.info("[send_sticker ok] id={}", sticker_id)
        return f"已发送 {sticker_id}"
