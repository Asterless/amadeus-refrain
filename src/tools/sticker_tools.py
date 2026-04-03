"""Sticker tools: SaveStickerTool and SendStickerTool for sticker library management."""

from pathlib import Path
from typing import Any

from loguru import logger

from src.sticker.store import StickerStore
from src.tools.base import Tool
from src.tools.context import ToolContext


class SaveStickerTool(Tool):
    """Save an image to the sticker library."""

    def __init__(self, store: StickerStore, superusers: set[str]) -> None:
        self._store = store
        self._superusers = superusers

    @property
    def name(self) -> str:
        return "save_sticker"

    @property
    def description(self) -> str:
        return "收录一张表情包到你的表情包库。只在你完全理解图片含义、清楚使用场景、且符合自己性格时才调用。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_ref": {
                    "type": "string",
                    "description": "图片在磁盘上的路径（来自 image_ref 块）",
                },
                "description": {
                    "type": "string",
                    "description": "表情包内容描述",
                },
                "usage_hint": {
                    "type": "string",
                    "description": "适合使用该表情包的场景说明",
                },
            },
            "required": ["image_ref", "description", "usage_hint"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        image_ref: str = kwargs["image_ref"]
        description: str = kwargs["description"]
        usage_hint: str = kwargs["usage_hint"]

        path = Path(image_ref)
        if not path.exists():
            return f"图片文件不存在: {image_ref}"

        image_data = path.read_bytes()
        source = "admin" if ctx.user_id in self._superusers else "auto"

        try:
            stk_id, is_new = self._store.add(image_data, description, usage_hint, source)
        except ValueError as e:
            return f"无法收录: {e}"

        if not is_new:
            return f"表情包已存在: {stk_id}"
        return f"{stk_id} 已收录"


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

        try:
            if ctx.group_id:
                await ctx.bot.send_group_msg(group_id=int(ctx.group_id), message=img_seg)
            else:
                await ctx.bot.send_private_msg(user_id=int(ctx.user_id), message=img_seg)
        except Exception as e:
            logger.error("send_sticker failed for {}: {}", sticker_id, e)
            return f"发送失败: {sticker_id}"

        self._store.record_send(sticker_id)
        return f"已发送表情包 {sticker_id}"
