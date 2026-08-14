"""LLM tools for NetEase Cloud Music."""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.music.client import NeteaseMusicClient
from src.tools.base import Tool
from src.tools.context import ToolContext


class MusicSearchTool(Tool):
    def __init__(self, client: NeteaseMusicClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "music_search"

    @property
    def description(self) -> str:
        return "搜索网易云音乐歌曲，返回歌曲 id、歌名、歌手和专辑。用户点歌或分享歌曲时先调用。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "歌名、歌手或搜索关键词"},
                "limit": {"type": "integer", "description": "结果数量，默认 8，最多 20"},
            },
            "required": ["keyword"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        keyword = str(kwargs.get("keyword") or "").strip()
        if not keyword:
            return "请提供歌名、歌手或关键词。"
        try:
            rows = await self._client.search(keyword, int(kwargs.get("limit", 8)))
        except Exception:
            logger.warning("netease search failed | keyword={}", keyword, exc_info=True)
            return "网易云搜索失败，请检查音乐 API 服务是否在线。"
        if not rows:
            return "没有找到匹配歌曲。"
        return "\n".join(
            f"{index}. {track.label} | 专辑={track.album} | song_id={track.id}"
            for index, track in enumerate(rows, 1)
        )


class MusicShareTool(Tool):
    def __init__(self, client: NeteaseMusicClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "music_share"

    @property
    def description(self) -> str:
        return "把网易云歌曲卡片分享到当前群或私聊；song_id 必须来自本轮 music_search 结果。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "song_id": {"type": "integer", "description": "网易云歌曲 id"},
                "title": {"type": "string", "description": "搜索结果中的歌名，仅用于成功提示"},
            },
            "required": ["song_id"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        if not ctx.bot or (not ctx.group_id and not ctx.user_id):
            return "没有可分享的聊天目标。"
        song_id = int(kwargs.get("song_id", 0))
        if song_id <= 0:
            return "歌曲 id 无效，请先调用 music_search。"

        try:
            track = await self._client.get_song(song_id)
        except Exception:
            return "无法验证歌曲，请稍后再试。"
        if track is None:
            return "网易云中不存在这首歌，请重新调用 music_search。"

        from nonebot.adapters.onebot.v11 import MessageSegment

        segment = MessageSegment.music("163", song_id)
        try:
            if ctx.group_id:
                await ctx.bot.send_group_msg(group_id=int(ctx.group_id), message=segment)
            else:
                await ctx.bot.send_private_msg(user_id=int(ctx.user_id), message=segment)
        except Exception:
            logger.warning("netease share failed | song_id={}", song_id, exc_info=True)
            return "歌曲分享失败，请确认 OneBot 实现支持网易云音乐消息段。"
        return f"已分享网易云歌曲 {song_id}：{track.label}"


class MusicQrLoginTool(Tool):
    def __init__(self, client: NeteaseMusicClient, superusers: set[str]) -> None:
        self._client = client
        self._superusers = superusers

    @property
    def name(self) -> str:
        return "music_login_qr"

    @property
    def description(self) -> str:
        return "处理管理员私聊命令 /登录网易云，发送网易云登录二维码；不要在群聊调用。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        if ctx.user_id not in self._superusers:
            return "只有管理员可以登录网易云。"
        if ctx.group_id or not ctx.bot:
            return "为避免登录二维码被他人扫描，请私聊我登录。"
        try:
            qrimg = await self._client.login_qr(ctx.user_id)
            from nonebot.adapters.onebot.v11 import MessageSegment

            await ctx.bot.send_private_msg(
                user_id=int(ctx.user_id),
                message=MessageSegment.image(self._client.qr_bytes(qrimg)),
            )
            return "登录二维码已发送。请用网易云音乐扫码确认，然后让我检查登录状态。"
        except Exception:
            logger.warning("netease qr login failed", exc_info=True)
            return "网易云二维码生成失败，请检查音乐 API 服务。"


class MusicLoginStatusTool(Tool):
    def __init__(self, client: NeteaseMusicClient, superusers: set[str]) -> None:
        self._client = client
        self._superusers = superusers

    @property
    def name(self) -> str:
        return "music_login_status"

    @property
    def description(self) -> str:
        return "处理管理员私聊命令 /检查登录状态，检查网易云二维码登录状态。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        if ctx.user_id not in self._superusers:
            return "只有管理员可以检查网易云登录。"
        if ctx.group_id:
            return "请在私聊中检查登录状态。"
        try:
            status = await self._client.check_qr(ctx.user_id)
        except Exception:
            logger.warning("netease qr status failed", exc_info=True)
            return "登录状态检查失败。"
        if status == "missing" and await self._client.is_logged_in():
            return "网易云当前已登录。"
        return {
            "logged_in": "网易云已登录，登录态已保存。",
            "expired": "二维码已过期，请重新生成。",
            "scanned": "已扫码，请在手机上确认登录。",
            "waiting": "等待扫码。",
            "missing": "没有待确认的二维码，请先生成登录二维码。",
        }.get(status, "登录状态未知。")
