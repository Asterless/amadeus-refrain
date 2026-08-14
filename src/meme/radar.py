"""Periodic multi-platform trend refresh."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from loguru import logger

from src.meme.store import MemeStore, TrendItem


class TrendProvider(Protocol):
    async def fetch(self, platform: str, limit: int) -> list[TrendItem]: ...

    async def close(self) -> None: ...


class UapiTrendProvider:
    """Fetch normalized chart entries from a UAPI-compatible hotboard endpoint."""

    def __init__(self, endpoint: str, *, timeout_seconds: float = 15.0) -> None:
        self._endpoint = endpoint
        self._client = httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True)

    async def fetch(self, platform: str, limit: int) -> list[TrendItem]:
        response = await self._client.get(self._endpoint, params={"type": platform})
        response.raise_for_status()
        payload: Any = response.json()
        rows: Any = payload.get("list") if isinstance(payload, dict) else None
        if rows is None and isinstance(payload, dict):
            data = payload.get("data")
            rows = data.get("list") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("hotboard response does not contain a list")

        result: list[TrendItem] = []
        for fallback_rank, row in enumerate(rows[:limit], 1):
            if not isinstance(row, dict):
                continue
            title = row.get("title") or row.get("name") or row.get("word")
            if not title:
                continue
            raw_rank = row.get("index") or row.get("rank") or fallback_rank
            try:
                rank = max(1, int(raw_rank))
            except (TypeError, ValueError):
                rank = fallback_rank
            result.append(
                TrendItem(
                    platform=platform,
                    title=str(title),
                    url=str(row.get("url") or row.get("link") or ""),
                    rank=rank,
                    hot_value=str(row.get("hot_value") or row.get("hot") or row.get("heat") or ""),
                )
            )
        return result

    async def close(self) -> None:
        await self._client.aclose()


class MemeRadar:
    """Refresh trend candidates in the background without blocking chat startup."""

    def __init__(
        self,
        store: MemeStore,
        provider: TrendProvider,
        *,
        platforms: list[str],
        refresh_minutes: int = 15,
        per_platform_limit: int = 30,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        self._provider = provider
        self._platforms = list(dict.fromkeys(p.casefold() for p in platforms if p.strip()))
        self._refresh_seconds = max(60, refresh_minutes * 60)
        self._per_platform_limit = per_platform_limit
        self._on_change = on_change
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def refresh_once(self) -> int:
        results = await asyncio.gather(
            *(self._provider.fetch(platform, self._per_platform_limit) for platform in self._platforms),
            return_exceptions=True,
        )
        items: list[TrendItem] = []
        for platform, result in zip(self._platforms, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("meme radar fetch failed | platform={} error={}", platform, result)
                continue
            items.extend(result)
        before_prompt = self._store.format_prompt_view()
        if items and self._store.update(items):
            prompt_changed = before_prompt != self._store.format_prompt_view()
            if prompt_changed and self._on_change:
                self._on_change()
            logger.info(
                "meme radar refreshed | fetched={} stored={} prompt_changed={}",
                len(items), self._store.count, prompt_changed,
            )
        return len(items)

    async def _run(self) -> None:
        try:
            while True:
                try:
                    await self.refresh_once()
                except Exception:
                    logger.exception("meme radar refresh failed")
                await asyncio.sleep(self._refresh_seconds)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        await self._provider.close()
