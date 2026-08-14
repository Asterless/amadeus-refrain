from nonebot.adapters.onebot.v11.exception import ActionFailed

from src.config import GroupConfig
from src.identity.models import Identity
from src.llm.scheduler import GroupChatScheduler
from src.memory.group_timeline import GroupTimeline


class _IdentityManager:
    def resolve(self) -> Identity:
        return Identity(id="test", name="test", personality="test")


class _Bot:
    def __init__(self) -> None:
        self.calls = 0

    async def send_group_msg(self, **_kwargs) -> None:
        self.calls += 1
        raise ActionFailed(message="rejected")


async def test_group_send_stops_after_retry_limit(monkeypatch) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("src.llm.scheduler.asyncio.sleep", no_sleep)
    scheduler = GroupChatScheduler(
        llm=object(),  # type: ignore[arg-type]
        timeline=GroupTimeline(),
        identity_mgr=_IdentityManager(),  # type: ignore[arg-type]
        group_config=GroupConfig(),
    )
    bot = _Bot()
    scheduler.set_bot(bot)  # type: ignore[arg-type]

    await scheduler._send_to_group("9", "hello")
    assert bot.calls == 5
