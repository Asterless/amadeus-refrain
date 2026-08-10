from __future__ import annotations

import asyncio

from src.tools.context import ToolContext
from src.tools.imagegen_tools import GenerateImageTool, _pending_prompts
from src.tools.imagegen_usage import ImageGenQuota


class _FakeBot:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self._message_id = 100

    async def send_group_msg(self, **kwargs: object) -> dict[str, int]:
        self.sent.append(kwargs)
        self._message_id += 1
        return {"message_id": self._message_id}

    async def send_private_msg(self, **kwargs: object) -> dict[str, int]:
        return await self.send_group_msg(**kwargs)


def _make_tool(tmp_path, rewriter=None) -> GenerateImageTool:
    return GenerateImageTool(
        base_url="https://example.invalid/v1",
        api_key="test",
        model="test-image",
        size="1024x1024",
        timeout_seconds=30,
        max_prompt_chars=500,
        daily_global_limit=10,
        daily_user_limit=5,
        daily_group_limit=5,
        usage_file=str(tmp_path / "usage.json"),
        prompt_rewriter=rewriter,
    )


async def test_revision_is_semantically_rewritten_without_conflict(tmp_path) -> None:
    async def rewrite(current: str, revision: str) -> str:
        assert "红围巾" in current
        assert revision == "把红围巾改成蓝色"
        return "一只戴蓝色围巾的柴犬站在海边"

    _pending_prompts.clear()
    bot = _FakeBot()
    tool = _make_tool(tmp_path, rewrite)
    await tool._request_confirmation(
        ToolContext(bot=bot, user_id="1", group_id="9"), "一只戴红围巾的柴犬站在海边", "1", "9",
    )

    assert await tool.try_confirm(bot=bot, user_id="1", group_id="9", text="把红围巾改成蓝色")
    prompt = _pending_prompts[("1", "9")].effective_prompt()
    assert "蓝色围巾" in prompt
    assert "红围巾" not in prompt
    assert "修改要求" not in prompt


async def test_unrelated_text_is_not_consumed_but_reply_is(tmp_path) -> None:
    async def rewrite(_current: str, _revision: str) -> str:
        return "新提示词"

    _pending_prompts.clear()
    bot = _FakeBot()
    tool = _make_tool(tmp_path, rewrite)
    await tool._request_confirmation(
        ToolContext(bot=bot, user_id="1", group_id="9"), "一只猫", "1", "9",
    )
    pending = _pending_prompts[("1", "9")]

    assert not tool.can_handle_confirmation("1", "9", "今天吃什么")
    assert not await tool.try_confirm(bot=bot, user_id="1", group_id="9", text="今天吃什么")
    assert tool.can_handle_confirmation("1", "9", "", pending.confirmation_message_id)
    assert await tool.try_confirm(
        bot=bot, user_id="1", group_id="9", text="",
        reply_message_id=pending.confirmation_message_id,
    )
    assert any("还没收到确认" in str(message) for message in bot.sent)


async def test_personal_and_proactive_pending_are_isolated(tmp_path) -> None:
    _pending_prompts.clear()
    bot = _FakeBot()
    tool = _make_tool(tmp_path)
    await tool._request_confirmation(
        ToolContext(bot=bot, user_id="1", group_id="9"), "个人请求", "1", "9",
    )
    await tool._request_confirmation(
        ToolContext(bot=bot, user_id="", group_id="9"), "主动请求", "", "9",
    )

    assert tool.cancel_pending("1", "9")
    assert ("1", "9") not in _pending_prompts
    assert ("", "9") in _pending_prompts
    assert tool.has_pending("2", "9")


async def test_quota_reservation_is_atomic(tmp_path) -> None:
    quota = ImageGenQuota(str(tmp_path / "usage.json"))

    async def reserve():
        return await quota.reserve(
            user_id="u", group_id="g", global_limit=1, user_limit=1,
            group_limit=1, cooldown_s=0,
        )

    first, second = await asyncio.gather(reserve(), reserve())
    reservations = [result[0] for result in (first, second) if result[0] is not None]
    assert len(reservations) == 1
    await quota.commit(reservations[0])
    stats = quota._load()["dates"][quota._today()]
    assert stats["global"] == 1


async def test_failed_generation_releases_reservation(tmp_path) -> None:
    _pending_prompts.clear()
    bot = _FakeBot()
    tool = _make_tool(tmp_path)

    async def fail(_prompt: str) -> bytes:
        raise RuntimeError("failed")

    tool._generate = fail  # type: ignore[method-assign]
    await tool._request_confirmation(
        ToolContext(bot=bot, user_id="1", group_id="9"), "一只猫", "1", "9",
    )
    assert await tool.try_confirm(bot=bot, user_id="1", group_id="9", text="确认")
    assert not tool.usage._reservations
    assert not tool.usage._load().get("dates")
