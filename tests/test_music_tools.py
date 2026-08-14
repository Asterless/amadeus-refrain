"""Music tool permission and OneBot delivery tests."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.music.client import MusicTrack
from src.tools.context import ToolContext
from src.tools.music_tools import MusicLoginStatusTool, MusicQrLoginTool, MusicSearchTool, MusicShareTool


class FakeMusicClient:
    async def search(self, keyword: str, limit: int = 8) -> list[MusicTrack]:
        return [MusicTrack(id=123, name="红莲华", artists="LiSA", album="LEO-NiNE")]

    async def login_qr(self, session_key: str) -> str:
        return "data:image/png;base64,aW1hZ2U="

    async def get_song(self, song_id: int) -> MusicTrack | None:
        if song_id != 123:
            return None
        return MusicTrack(id=123, name="红莲华", artists="LiSA", album="LEO-NiNE")

    async def check_qr(self, session_key: str) -> str:
        return "missing"

    async def is_logged_in(self) -> bool:
        return True

    @staticmethod
    def qr_bytes(qrimg: str) -> bytes:
        return b"image"


async def test_music_search_returns_ids_for_followup() -> None:
    tool = MusicSearchTool(FakeMusicClient())  # type: ignore[arg-type]
    result = await tool.execute(ToolContext(user_id="1"), keyword="红莲华")

    assert "红莲华 - LiSA" in result
    assert "song_id=123" in result


async def test_music_share_sends_onebot_netease_card() -> None:
    bot = MagicMock()
    bot.send_group_msg = AsyncMock()
    tool = MusicShareTool(FakeMusicClient())  # type: ignore[arg-type]
    segment = MagicMock()

    with patch("nonebot.adapters.onebot.v11.MessageSegment.music", return_value=segment) as music:
        result = await tool.execute(
            ToolContext(bot=bot, user_id="1", group_id="100"),
            song_id=123,
            title="红莲华",
        )

    music.assert_called_once_with("163", 123)
    bot.send_group_msg.assert_awaited_once_with(group_id=100, message=segment)
    assert "已分享" in result


async def test_music_share_rejects_mismatched_search_id() -> None:
    bot = MagicMock()
    bot.send_group_msg = AsyncMock()
    result = await MusicShareTool(FakeMusicClient()).execute(  # type: ignore[arg-type]
        ToolContext(bot=bot, user_id="1", group_id="100"), song_id=999, title="红莲华"
    )

    assert "不存在" in result
    bot.send_group_msg.assert_not_awaited()


async def test_qr_login_requires_admin_private_chat() -> None:
    client = FakeMusicClient()
    tool = MusicQrLoginTool(client, {"1"})  # type: ignore[arg-type]
    bot = MagicMock()
    bot.send_private_msg = AsyncMock()

    denied = await tool.execute(ToolContext(bot=bot, user_id="2"))
    group_denied = await tool.execute(ToolContext(bot=bot, user_id="1", group_id="100"))
    assert "只有管理员" in denied
    assert "请私聊" in group_denied

    with patch("nonebot.adapters.onebot.v11.MessageSegment.image", return_value="qr"):
        allowed = await tool.execute(ToolContext(bot=bot, user_id="1"))
    bot.send_private_msg.assert_awaited_once_with(user_id=1, message="qr")
    assert "二维码已发送" in allowed


async def test_login_status_detects_persisted_login_without_pending_qr() -> None:
    tool = MusicLoginStatusTool(FakeMusicClient(), {"1"})  # type: ignore[arg-type]
    result = await tool.execute(ToolContext(user_id="1"))

    assert result == "网易云当前已登录。"


def test_music_commands_are_documented_as_slash_commands() -> None:
    source = Path("src/plugins/chat/__init__.py").read_text(encoding="utf-8")
    assert '"/登录网易云"' in source
    assert '"/检查登录状态"' in source
    assert 'command in {"登录网易云"' not in source
