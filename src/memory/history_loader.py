"""启动时从 NapCat HTTP API 拉取群历史消息，填充群聊上下文。"""

from typing import Any

import aiohttp
from loguru import logger

from src.memory.group_timeline import GroupTimeline


async def load_group_history(
    napcat_url: str,
    group_ids: list[str],
    timeline: GroupTimeline,
    count: int = 30,
    bot_self_id: str = "",
) -> None:
    """从 NapCat 拉取多个群的历史消息。"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        for gid in group_ids:
            try:
                await _load_one_group(session, napcat_url, gid, timeline, count, bot_self_id)
            except Exception:
                logger.warning("load_history failed | group={}", gid, exc_info=True)


async def _load_one_group(
    session: aiohttp.ClientSession,
    napcat_url: str,
    group_id: str,
    timeline: GroupTimeline,
    count: int,
    bot_self_id: str = "",
) -> None:
    async with session.post(
        f"{napcat_url}/get_group_msg_history",
        json={"group_id": int(group_id), "count": count},
    ) as resp:
        data: dict[str, Any] = await resp.json()

    if data.get("retcode") != 0:
        logger.warning("get_group_msg_history error | group={} resp={}", group_id, data.get("message", ""))
        return

    messages: list[dict[str, Any]] = data.get("data", {}).get("messages", [])
    if not messages:
        return

    loaded = 0

    for msg in messages:
        sender: dict[str, Any] = msg.get("sender", {})
        user_id = str(sender.get("user_id", ""))
        nickname = sender.get("nickname", "") or sender.get("card", "") or user_id

        # 提取纯文本
        text_parts: list[str] = []
        for seg in msg.get("message", []):
            if seg.get("type") == "text":
                text_parts.append(seg.get("data", {}).get("text", ""))
        text = "".join(text_parts).strip()
        if not text:
            continue

        # Bot's own messages → role="assistant"; others → role="user" with speaker tag
        if bot_self_id and user_id == bot_self_id:
            timeline.add(group_id, role="assistant", content=text)
        else:
            timeline.add(group_id, role="user", speaker=f"{nickname}({user_id})", content=text)
        loaded += 1

    logger.info("history loaded | group={} messages={}", group_id, loaded)
