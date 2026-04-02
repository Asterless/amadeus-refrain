"""Memo tools: RecallMemoTool and UpdateMemoTool for structured memo retrieval and update."""

import asyncio
from typing import Any, Literal

from loguru import logger

from src.memory.memo_store import MemoStore
from src.tools.base import Tool
from src.tools.context import ToolContext


class RecallMemoTool(Tool):
    """Retrieve memos by exact ID or fuzzy query."""

    def __init__(self, store: MemoStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "recall_memo"

    @property
    def description(self) -> str:
        return (
            "查询用户或群组的 memo 档案。"
            "用 id（如 user_123456 / group_987654）精确查询完整内容；"
            "用 query 模糊搜索昵称或关键词，返回摘要列表（含 QQ 号）供进一步精确查询。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "精确 memo ID，如 user_123456 或 group_987654",
                },
                "query": {
                    "type": "string",
                    "description": "模糊搜索关键词，匹配 identity 字段或正文",
                },
                "kind": {
                    "type": "string",
                    "enum": ["user", "group"],
                    "description": "限制搜索范围，仅在使用 query 时生效",
                },
            },
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        memo_id: str | None = kwargs.get("id")
        query: str | None = kwargs.get("query")
        kind: Literal["user", "group"] | None = kwargs.get("kind")

        if memo_id is not None:
            memo = self._store.read(memo_id)
            if memo is None:
                return f"未找到 memo: {memo_id}"
            return memo.body

        if query is not None:
            query_lower = query.lower()
            ids_to_search = self._store.list_ids(kind=kind)
            results: list[str] = []
            for mid in ids_to_search:
                memo = self._store.read(mid)
                if memo is None:
                    continue
                if query_lower in memo.identity.lower() or query_lower in memo.body.lower():
                    results.append(f"{mid}: {memo.identity}")
            if not results:
                return f"未找到匹配 '{query}' 的 memo。"
            return "\n".join(results)

        return "请提供 id 或 query 参数。"


class UpdateMemoTool(Tool):
    """Overwrite a memo asynchronously (fire-and-forget)."""

    def __init__(self, store: MemoStore) -> None:
        self._store = store
        # Hold strong refs to background tasks to prevent premature GC
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def name(self) -> str:
        return "update_memo"

    @property
    def description(self) -> str:
        return "完整覆写某个用户或群组的 memo 档案内容。立即返回，后台异步写入。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "目标 memo ID，如 user_QQ号 或 group_群号",
                },
                "memo": {
                    "type": "string",
                    "description": "完整的 memo 新内容（全量覆写）",
                },
            },
            "required": ["id", "memo"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        memo_id: str = kwargs["id"]
        memo_content: str = kwargs["memo"]
        session_id: str = getattr(ctx, "session_id", "unknown")
        source = f"tool:{session_id}"
        task = asyncio.create_task(self._store.write(memo_id, memo_content, source))
        self._background_tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return "已提交更新"

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            logger.warning("update_memo task cancelled")
        elif exc := task.exception():
            logger.error("update_memo background write failed: {}", exc)
