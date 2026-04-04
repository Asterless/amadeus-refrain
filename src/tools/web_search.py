"""网页搜索工具：通过 DuckDuckGo 搜索互联网。"""

import warnings
from typing import Any

from src.tools.base import Tool
from src.tools.context import ToolContext

MAX_RESULTS = 5


class WebSearchTool(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "使用 DuckDuckGo 搜索互联网，返回相关网页的标题、链接和摘要。"
            "适合查询实时信息、新闻、技术文档等。"
            "如果需要查看某个结果的完整内容，可以用 web_fetch 抓取对应链接。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认 5，最多 10",
                },
            },
            "required": ["query"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        max_results = min(int(kwargs.get("max_results", MAX_RESULTS)), 10)

        try:
            results = await _ddg_search(query, max_results)
        except Exception as e:
            return f"搜索失败: {e}"

        if not results:
            return "未找到相关结果。"

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}")
        return "\n\n".join(lines)


async def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Run blocking DDGS.text() in a thread to avoid blocking the event loop."""
    import asyncio

    return await asyncio.to_thread(_ddg_search_sync, query, max_results)


def _ddg_search_sync(query: str, max_results: int) -> list[dict[str, str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from duckduckgo_search import DDGS

        d = DDGS()
    return d.text(query, max_results=max_results)  # type: ignore[return-value]
