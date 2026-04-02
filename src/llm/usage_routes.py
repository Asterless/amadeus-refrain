"""FastAPI routes for LLM usage stats."""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.llm.usage import UsageTracker


def create_usage_router(tracker: UsageTracker) -> APIRouter:
    router = APIRouter(prefix="/api/usage", tags=["usage"])

    @router.get("/today")
    async def today():
        return await tracker.summary_today()

    @router.get("/month")
    async def month(month: str | None = Query(None, description="YYYY-MM")):
        return await tracker.summary_month(month)

    @router.get("/top-users")
    async def top_users(days: int = Query(7), limit: int = Query(10)):
        return await tracker.top_users(days=days, limit=limit)

    @router.get("/top-groups")
    async def top_groups(days: int = Query(7), limit: int = Query(10)):
        return await tracker.top_groups(days=days, limit=limit)

    return router
