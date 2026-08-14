"""Smoke test for the image generation confirmation flow (no real API calls)."""

from __future__ import annotations

import asyncio
import pathlib
import tempfile
import time

from src.tools.context import ToolContext
from src.tools.imagegen_tools import EditImageTool, GenerateImageTool, _pending_prompts


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []

    async def send_group_msg(self, **kwargs: object) -> None:
        self.sent.append(("group", kwargs))

    async def send_private_msg(self, **kwargs: object) -> None:
        self.sent.append(("private", kwargs))


async def fake_rewrite(current: str, revision: str) -> str:
    if "把背景改成海边" in revision:
        return f"{current}，背景为海边"
    if "把红围巾改成蓝色" in revision:
        return current.replace("红围巾", "蓝色围巾")
    return f"{current}，{revision}"


def make_tool(usage_file: str) -> tuple[GenerateImageTool, list[str]]:
    tool = GenerateImageTool(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="gpt-image-2",
        size="1024x1024",
        timeout_seconds=30,
        max_prompt_chars=500,
        daily_global_limit=10,
        daily_user_limit=5,
        daily_group_limit=5,
        cooldown_seconds=0,
        usage_file=usage_file,
        prompt_rewriter=fake_rewrite,
    )

    generated_prompts: list[str] = []

    async def fake_generate(prompt: str) -> bytes:
        generated_prompts.append(prompt)
        return b"fakejpeg"

    tool._generate = fake_generate  # type: ignore[method-assign]
    return tool, generated_prompts


def make_edit_tool(usage: object) -> tuple[EditImageTool, list[str]]:
    tool = EditImageTool(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="gpt-image-2",
        size="1024x1024",
        timeout_seconds=30,
        max_prompt_chars=500,
        daily_global_limit=10,
        daily_user_limit=5,
        daily_group_limit=5,
        cooldown_seconds=0,
        usage=usage,  # type: ignore[arg-type]
        prompt_rewriter=fake_rewrite,
    )

    edited_prompts: list[str] = []

    async def fake_edits(_image_bytes: bytes, prompt: str) -> bytes:
        edited_prompts.append(prompt)
        return b"editedimg"

    tool._call_edits = fake_edits  # type: ignore[method-assign]
    return tool, edited_prompts


async def main() -> None:
    tmp = pathlib.Path(tempfile.gettempdir()) / "imagegen_confirm_test.json"
    tmp.unlink(missing_ok=True)
    tool, generated_prompts = make_tool(str(tmp))
    _pending_prompts.clear()
    bot = FakeBot()

    # 初始没有待确认请求
    assert not tool.has_pending("1001", "9001")
    assert not await tool.try_confirm(bot=bot, user_id="1001", group_id="9001", text="确认")

    # 请求确认
    ctx = ToolContext(bot=bot, user_id="1001", group_id="9001")
    result = await tool._request_confirmation(ctx, "一只戴红围巾的柴犬", "1001", "9001")
    assert "确认" in result
    assert tool.has_pending("1001", "9001")
    assert any("柴犬" in str(m) for _, m in bot.sent)
    assert any("/取消生图" in str(m) for _, m in bot.sent)

    # 修改描述 -> 重新确认
    assert await tool.try_confirm(bot=bot, user_id="1001", group_id="9001", text="把背景改成海边")
    assert tool.has_pending("1001", "9001")
    assert any("海边" in str(m) for _, m in bot.sent)
    effective = _pending_prompts[("1001", "9001")].effective_prompt()
    assert "一只戴红围巾的柴犬" in effective
    assert "背景为海边" in effective

    # 连续修改继续累加，并明确以后一次要求为准
    assert await tool.try_confirm(bot=bot, user_id="1001", group_id="9001", text="把红围巾改成蓝色")
    effective = _pending_prompts[("1001", "9001")].effective_prompt()
    assert "背景为海边" in effective
    assert "蓝色围巾" in effective
    assert "红围巾" not in effective

    # 确认 -> 生成并发送图片
    assert await tool.try_confirm(bot=bot, user_id="1001", group_id="9001", text="ok")
    assert not tool.has_pending("1001", "9001")
    assert generated_prompts == [effective]
    assert any("CQ:image" in str(m.get("message")) for _, m in bot.sent)

    # 取消
    ctx2 = ToolContext(bot=bot, user_id="1002", group_id="9001")
    await tool._request_confirmation(ctx2, "一只猫", "1002", "9001")
    assert await tool.try_confirm(bot=bot, user_id="1002", group_id="9001", text="取消")
    assert not tool.has_pending("1002", "9001")
    assert any("已取消" in str(m.get("message")) for _, m in bot.sent)

    # 取消命令和普通取消走同一套状态清理，不消耗额度
    usage_before_cancel = tool.usage._load()
    ctx_command_cancel = ToolContext(bot=bot, user_id="1004", group_id="9001")
    await tool._request_confirmation(ctx_command_cancel, "一只兔子", "1004", "9001")
    assert await tool.try_confirm(bot=bot, user_id="1004", group_id="9001", text="/取消生图")
    assert not tool.has_pending("1004", "9001")
    assert tool.usage._load() == usage_before_cancel
    assert not tool.cancel_pending("1004", "9001")

    # 主动插话的待确认请求也可以被群内成员取消
    ctx_proactive_cancel = ToolContext(bot=bot, user_id="", group_id="9002")
    await tool._request_confirmation(ctx_proactive_cancel, "一张夜景", "", "9002")
    assert tool.cancel_pending("1010", "9002")
    assert not tool.has_pending("1010", "9002")

    # 过期
    ctx3 = ToolContext(bot=bot, user_id="1003", group_id="9001")
    await tool._request_confirmation(ctx3, "一只狗", "1003", "9001")
    _pending_prompts[("1003", "9001")].created_at = time.time() - 300  # type: ignore[union-attr]
    assert await tool.try_confirm(bot=bot, user_id="1003", group_id="9001", text="确认")
    assert not tool.has_pending("1003", "9001")
    assert any("已过期" in str(m.get("message")) for _, m in bot.sent)

    # 主动插话（无指定用户）：群内任意成员可确认
    ctx4 = ToolContext(bot=bot, user_id="", group_id="9001")
    await tool._request_confirmation(ctx4, "一张风景照", "", "9001")
    assert tool.has_pending("1009", "9001")
    assert await tool.try_confirm(bot=bot, user_id="1009", group_id="9001", text="确认")
    assert not tool.has_pending("1009", "9001")

    # 额度计数：确认两次后应为 2
    assert int(tool.usage._load().get("dates", {}).get(tool.usage._today(), {}).get("global", 0)) == 2

    # 图生图：execute 定位图片标签 -> 确认 -> 出图
    edit_tool, edited_prompts = make_edit_tool(tool.usage)
    fake_img = pathlib.Path(tempfile.gettempdir()) / "imagegen_edit_input.jpg"
    fake_img.write_bytes(b"fakeinput")
    ctx_edit = ToolContext(
        bot=bot, user_id="1005", group_id="9001",
        extra={"image_tags": {"img:1": str(fake_img)}},
    )
    result = await edit_tool.execute(ctx_edit, prompt="把背景改成海边", image="img:1")
    assert "确认" in result
    assert edit_tool.has_pending("1005", "9001")
    assert await edit_tool.try_confirm(bot=bot, user_id="1005", group_id="9001", text="再加一轮落日")
    assert await edit_tool.try_confirm(bot=bot, user_id="1005", group_id="9001", text="确认")
    assert not edit_tool.has_pending("1005", "9001")
    assert "把背景改成海边" in edited_prompts[0]
    assert "再加一轮落日" in edited_prompts[0]
    assert any("CQ:image" in str(m.get("message")) for _, m in bot.sent)

    # 图生图：没有图片标签 -> 返回提示而不是调用
    ctx_no_img = ToolContext(bot=bot, user_id="1006", group_id="9001")
    result2 = await edit_tool.execute(ctx_no_img, prompt="改一下")
    assert "图片" in result2
    fake_img.unlink(missing_ok=True)

    _pending_prompts.clear()
    tmp.unlink(missing_ok=True)
    print("ALL CONFIRM TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
