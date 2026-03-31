"""长期记忆：用户画像 + 关键事件，.qmd 文件持久化。

每个用户一个文件：data/memories/{user_id}.qmd

格式：
    ## 画像

    - nickname: 小明
    - 爱好: 编程
    - 性格: 开朗

    ## 事件

    - 2025-03-31 14:30: 提到自己是程序员
    - 2025-03-31 15:00: 说最近在学 Rust
"""

import asyncio
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from zoneinfo import ZoneInfo

import aiofiles

_TRAIT_RE = re.compile(r"^-\s+(.+?):\s*(.+)$")
_EVENT_RE = re.compile(r"^-\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}):\s*(.+)$")
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")


class UserProfile(TypedDict):
    nickname: str
    traits: dict[str, str]


class EventRecord(TypedDict):
    content: str
    timestamp: float


class LongTermMemory:
    def __init__(self, memory_dir: str = "data/memories") -> None:
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _path(self, user_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", user_id) if not _SAFE_ID_RE.match(user_id) else user_id
        return self._dir / f"{safe_id}.qmd"

    async def _read(self, user_id: str) -> str:
        p = self._path(user_id)
        if not p.exists():
            return ""
        async with aiofiles.open(p, encoding="utf-8") as f:
            return await f.read()

    async def _write(self, user_id: str, content: str) -> None:
        async with aiofiles.open(self._path(user_id), "w", encoding="utf-8") as f:
            await f.write(content)

    # ── 解析 ──

    def _parse(self, text: str) -> tuple[UserProfile, list[EventRecord]]:
        profile = UserProfile(nickname="", traits={})
        events: list[EventRecord] = []

        section = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                section = stripped[3:].strip()
                continue
            if not stripped or stripped.startswith("#"):
                continue

            if section == "画像":
                m = _TRAIT_RE.match(stripped)
                if m:
                    key, val = m.group(1), m.group(2)
                    if key == "nickname":
                        profile["nickname"] = val
                    else:
                        profile["traits"][key] = val

            elif section == "事件":
                m = _EVENT_RE.match(stripped)
                if m:
                    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=TZ_SHANGHAI)
                    events.append(EventRecord(content=m.group(2), timestamp=dt.timestamp()))

        return profile, events

    def _serialize(self, profile: UserProfile, events: list[EventRecord]) -> str:
        lines: list[str] = ["## 画像", ""]
        if profile["nickname"]:
            lines.append(f"- nickname: {profile['nickname']}")
        for k, v in profile["traits"].items():
            lines.append(f"- {k}: {v}")

        lines.extend(["", "## 事件", ""])
        for e in events:
            dt = datetime.fromtimestamp(e["timestamp"], tz=TZ_SHANGHAI).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- {dt}: {e['content']}")

        return "\n".join(lines) + "\n"

    # ── 用户画像 ──

    async def get_profile(self, user_id: str) -> UserProfile | None:
        text = await self._read(user_id)
        if not text:
            return None
        profile, _ = self._parse(text)
        if not profile["nickname"] and not profile["traits"]:
            return None
        return profile

    async def update_profile(
        self, user_id: str, *, nickname: str | None = None, traits: dict[str, str] | None = None
    ) -> None:
        async with self._locks[user_id]:
            text = await self._read(user_id)
            profile, events = self._parse(text) if text else (UserProfile(nickname="", traits={}), [])

            if nickname:
                profile["nickname"] = nickname
            if traits:
                profile["traits"].update(traits)

            await self._write(user_id, self._serialize(profile, events))

    # ── 关键事件 ──

    async def add_event(self, user_id: str, content: str) -> None:
        async with self._locks[user_id]:
            text = await self._read(user_id)
            profile, events = self._parse(text) if text else (UserProfile(nickname="", traits={}), [])

            events.append(EventRecord(content=content, timestamp=time.time()))
            await self._write(user_id, self._serialize(profile, events))

    async def get_events(self, user_id: str, limit: int = 10) -> list[EventRecord]:
        text = await self._read(user_id)
        if not text:
            return []
        _, events = self._parse(text)
        return sorted(events, key=lambda e: e["timestamp"], reverse=True)[:limit]

    async def get_full_context(self, user_id: str) -> str:
        return await self._read(user_id)
