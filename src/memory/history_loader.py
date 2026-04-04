"""启动时从 NapCat HTTP API 拉取群历史消息，填充群聊上下文。"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from src.constants.qq_face import face_to_text
from src.memory.group_timeline import GroupTimeline
from src.memory.image_cache import ImageCache
from src.memory.types import Content, ContentBlock, ImageRefBlock, TextBlock
from src.sticker.store import StickerStore


async def load_group_history(
    napcat_url: str,
    group_ids: list[str],
    timeline: GroupTimeline,
    count: int = 30,
    bot_self_id: str = "",
    image_cache: ImageCache | None = None,
    sticker_store: StickerStore | None = None,
) -> None:
    """从 NapCat 拉取多个群的历史消息。"""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        for gid in group_ids:
            try:
                await _load_one_group(
                    session, napcat_url, gid, timeline, count, bot_self_id, image_cache, sticker_store
                )
            except Exception:
                logger.warning("load_history failed | group={}", gid, exc_info=True)


async def _load_one_group(
    session: aiohttp.ClientSession,
    napcat_url: str,
    group_id: str,
    timeline: GroupTimeline,
    count: int,
    bot_self_id: str = "",
    image_cache: ImageCache | None = None,
    sticker_store: StickerStore | None = None,
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

    t0 = time.perf_counter()
    loaded = 0
    self_count = 0

    for msg in messages:
        sender: dict[str, Any] = msg.get("sender", {})
        user_id = str(sender.get("user_id", ""))
        nickname = sender.get("nickname", "") or sender.get("card", "") or user_id

        content = await _extract_content(msg.get("message", []), session, image_cache, sticker_store)
        if not content:
            continue

        msg_id = msg.get("message_id")
        if bot_self_id and user_id == bot_self_id:
            timeline.add(group_id, role="assistant", content=content)
            self_count += 1
        else:
            timeline.add(
                group_id, role="user", speaker=f"{nickname}({user_id})",
                content=content, message_id=msg_id,
            )
        loaded += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "history loaded | group={} messages={} self={} bot_self_id={} elapsed={:.0f}ms",
        group_id, loaded, self_count, bot_self_id, elapsed_ms,
    )


async def _extract_content(
    segments: list[dict[str, Any]],
    session: aiohttp.ClientSession,
    image_cache: ImageCache | None,
    sticker_store: StickerStore | None = None,
) -> Content:
    """Extract text, face, and image segments into a Content value."""
    text_parts: list[str] = []
    # Collect image download coroutines for concurrent execution
    image_tasks: list[asyncio.Task[ImageRefBlock | None]] = []

    for seg in segments:
        seg_type = seg.get("type", "")
        seg_data: dict[str, Any] = seg.get("data", {})

        if seg_type == "text":
            text_parts.append(seg_data.get("text", ""))
        elif seg_type == "face":
            face_id = seg_data.get("id", "")
            try:
                text_parts.append(face_to_text(int(face_id)))
            except (ValueError, TypeError):
                text_parts.append("«表情»")
        elif seg_type == "image" and image_cache is not None:
            url = seg_data.get("url", "")
            file_id = seg_data.get("file", "")
            if url and file_id:
                file_id = file_id.split(".")[0] if "." in file_id else file_id
                image_tasks.append(
                    asyncio.ensure_future(image_cache.save(session, url=url, file_id=file_id))
                )
            else:
                text_parts.append("«图片»")
        elif seg_type == "image":
            text_parts.append("«图片»")

    # Resolve all image downloads concurrently
    images: list[ImageRefBlock] = []
    if image_tasks:
        t0 = time.perf_counter()
        results = await asyncio.gather(*image_tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException) or r is None:
                text_parts.append("«图片»")
            else:
                # Check if the downloaded image matches a known sticker
                if sticker_store is not None and image_cache is not None:
                    cached_path = Path(r["path"])
                    if cached_path.exists():
                        image_data = cached_path.read_bytes()
                        stk_id = sticker_store.lookup_by_hash(image_data)
                        if stk_id is not None:
                            sticker_path = sticker_store.resolve_path(stk_id)
                            if sticker_path is not None:
                                # Remove the duplicate from image_cache and use sticker path
                                cached_path.unlink(missing_ok=True)
                                images.append(
                                    ImageRefBlock(
                                        type="image_ref",
                                        path=str(sticker_path),
                                        media_type=r["media_type"],
                                    )
                                )
                                logger.debug(
                                    "history image matched sticker | sticker_id={}", stk_id
                                )
                                continue
                images.append(r)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "history image batch | tasks={} ok={} elapsed={:.0f}ms",
            len(image_tasks), len(images), elapsed_ms,
        )

    text = "".join(text_parts).strip()

    if not images:
        return text

    blocks: list[ContentBlock] = []
    if text:
        blocks.append(TextBlock(type="text", text=text))
    blocks.extend(images)
    return blocks
