"""Persistent daily quota tracking for image generation."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class ImageGenReservation:
    token: str
    user_id: str
    group_id: str | None


class ImageGenQuota:
    """Track daily image counts (global / per user / per group) and per-user cooldown.

    Counters persist to a JSON file so limits survive bot restarts.

    File layout::

        {
          "dates": {
            "2026-08-10": {
              "global": 3,
              "users": {"1718666182": 2},
              "groups": {"1022565855": 2}
            }
          },
          "cooldowns": {"u:1718666182": 1786374000.0}
        }
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()
        self._reservations: dict[str, ImageGenReservation] = {}

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d")

    def _load(self) -> dict[str, Any]:
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(self._path)

    @staticmethod
    def _day_stats(data: dict[str, Any], date: str) -> dict[str, Any]:
        return data.get("dates", {}).get(date, {"global": 0, "users": {}, "groups": {}})

    async def check(
        self,
        *,
        user_id: str,
        group_id: str | None,
        global_limit: int,
        user_limit: int,
        group_limit: int,
        cooldown_s: float,
    ) -> tuple[bool, str]:
        """Return (allowed, reason). Call before generating; count only on success."""
        async with self._lock:
            data = self._load()
            stats = self._day_stats(data, self._today())
            now = time.time()

            if cooldown_s > 0 and user_id:
                last = data.get("cooldowns", {}).get(f"u:{user_id}", 0.0)
                remain = last + cooldown_s - now
                if remain > 0:
                    return False, f"生图太频繁，请 {remain:.0f} 秒后再试"
            if global_limit > 0 and int(stats.get("global", 0)) >= global_limit:
                return False, f"今日全 bot 生图额度已用完（{stats.get('global', 0)}/{global_limit}）"
            if user_limit > 0 and user_id and int(stats.get("users", {}).get(user_id, 0)) >= user_limit:
                return False, (
                    f"你今天生图额度已用完"
                    f"（{stats.get('users', {}).get(user_id, 0)}/{user_limit}，明天 0 点恢复）"
                )
            if group_limit > 0 and group_id and int(stats.get("groups", {}).get(group_id, 0)) >= group_limit:
                return False, (
                    f"本群今日生图额度已用完"
                    f"（{stats.get('groups', {}).get(group_id, 0)}/{group_limit}，明天 0 点恢复）"
                )
            return True, ""

    async def reserve(
        self,
        *,
        user_id: str,
        group_id: str | None,
        global_limit: int,
        user_limit: int,
        group_limit: int,
        cooldown_s: float,
    ) -> tuple[ImageGenReservation | None, str]:
        """Atomically check limits and reserve one generation slot."""
        async with self._lock:
            data = self._load()
            stats = self._day_stats(data, self._today())
            now = time.time()
            active = tuple(self._reservations.values())

            if cooldown_s > 0 and user_id:
                last = data.get("cooldowns", {}).get(f"u:{user_id}", 0.0)
                if any(item.user_id == user_id for item in active):
                    return None, "已有一张图片正在生成，请等待完成后再试"
                remain = last + cooldown_s - now
                if remain > 0:
                    return None, f"生图太频繁，请 {remain:.0f} 秒后再试"

            global_used = int(stats.get("global", 0)) + len(active)
            user_used = int(stats.get("users", {}).get(user_id, 0)) + sum(
                item.user_id == user_id for item in active
            )
            group_used = int(stats.get("groups", {}).get(group_id, 0)) + sum(
                item.group_id == group_id for item in active
            )
            if global_limit > 0 and global_used >= global_limit:
                return None, f"今日全 bot 生图额度已用完（{global_used}/{global_limit}）"
            if user_limit > 0 and user_id and user_used >= user_limit:
                return None, f"你今天生图额度已用完（{user_used}/{user_limit}，明天 0 点恢复）"
            if group_limit > 0 and group_id and group_used >= group_limit:
                return None, f"本群今日生图额度已用完（{group_used}/{group_limit}，明天 0 点恢复）"

            reservation = ImageGenReservation(uuid.uuid4().hex, user_id, group_id)
            self._reservations[reservation.token] = reservation
            return reservation, ""

    async def commit(self, reservation: ImageGenReservation) -> None:
        """Persist a successful reserved generation."""
        async with self._lock:
            current = self._reservations.pop(reservation.token, None)
            if current is None:
                return
            self._record_unlocked(current.user_id, current.group_id)

    async def release(self, reservation: ImageGenReservation) -> None:
        """Release a failed or abandoned generation reservation."""
        async with self._lock:
            self._reservations.pop(reservation.token, None)

    async def record(self, *, user_id: str, group_id: str | None) -> None:
        """Increment counters after a successful generation."""
        async with self._lock:
            self._record_unlocked(user_id, group_id)

    def _record_unlocked(self, user_id: str, group_id: str | None) -> None:
        data = self._load()
        date = self._today()
        dates = data.setdefault("dates", {})
        day = dates.setdefault(date, {"global": 0, "users": {}, "groups": {}})
        day["global"] = int(day.get("global", 0)) + 1
        if user_id:
            users = day.setdefault("users", {})
            users[user_id] = int(users.get(user_id, 0)) + 1
            data.setdefault("cooldowns", {})[f"u:{user_id}"] = time.time()
        if group_id:
            groups = day.setdefault("groups", {})
            groups[group_id] = int(groups.get(group_id, 0)) + 1
        keep = sorted(dates.keys())[-7:]
        for old_date in list(dates.keys()):
            if old_date not in keep:
                del dates[old_date]
        self._save(data)

    async def summary(
        self,
        *,
        user_id: str,
        group_id: str | None,
        global_limit: int,
        user_limit: int,
        group_limit: int,
        cooldown_s: float,
    ) -> str:
        """Human-readable quota summary for the /生图额度 command."""
        async with self._lock:
            data = self._load()
            stats = self._day_stats(data, self._today())
            now = time.time()

            def fmt(used: int, limit: int) -> str:
                return f"{used}/{limit}" if limit > 0 else f"{used}/不限"

            lines = [
                f"今日生图额度：全 bot {fmt(int(stats.get('global', 0)), global_limit)}",
            ]
            if group_id:
                lines.append(f"本群 {fmt(int(stats.get('groups', {}).get(group_id, 0)), group_limit)}")
            if user_id:
                lines.append(f"你 {fmt(int(stats.get('users', {}).get(user_id, 0)), user_limit)}")
                last = data.get("cooldowns", {}).get(f"u:{user_id}", 0.0)
                remain = last + cooldown_s - now
                lines.append(f"冷却：{'立即可用' if remain <= 0 else f'{remain:.0f} 秒后可用'}")
            return "\n".join(lines)

    def reset_today(self) -> None:
        """Force-clear today's counters (admin escape hatch)."""
        try:
            data = self._load()
            data.get("dates", {}).pop(self._today(), None)
            self._save(data)
            logger.info("imagegen quota reset for today")
        except OSError:
            logger.warning("imagegen quota reset failed", exc_info=True)
