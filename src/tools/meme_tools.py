"""Tools for inspecting current trends and verifying meme meanings."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

from src.meme.store import MemeStore, TrendItem
from src.tools.base import Tool
from src.tools.context import ToolContext
from src.tools.web_search import SearchFunction, _ddg_search

_TERM_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]{2,}")
_GENERIC_TERMS = {"什么梗", "是什么梗", "网络用语", "意思", "来源", "梗", "meme"}
_MEME_CUES = ("梗", "meme", "表情包", "网络用语", "什么意思", "来源", "出处", "走红", "流行", "热词")
_SPAM_CUES = ("送彩金", "彩票平台", "棋牌", "投注", "博彩", "官方入口", "app下载")
_SOCIAL_HOSTS = ("bilibili.com", "douyin.com", "xiaohongshu.com", "weibo.com", "zhihu.com")
_QUERY_SUFFIX_RE = re.compile(r"(?:是什么|啥|什么)?(?:梗|meme|表情包|网络用语)+$", re.IGNORECASE)


def _normalize_query(query: str) -> str:
    cleaned = re.sub(r"[\s,，。！？?！#]+", " ", query).strip()
    previous = ""
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = _QUERY_SUFFIX_RE.sub("", cleaned).strip()
    return cleaned or query.strip()


def _format_trends(rows: list[TrendItem]) -> str:
    if not rows:
        return "当前梗库中没有匹配的实时热点。"
    lines: list[str] = []
    for item in rows:
        heat = f" | 热度 {item.hot_value}" if item.hot_value else ""
        link = f"\n   {item.url}" if item.url else ""
        lines.append(f"- [{item.platform}] #{item.rank} {item.title}{heat}{link}")
    return "\n".join(lines)


def _relevance_score(query: str, row: dict[str, str]) -> int:
    """Reject search-engine noise before exposing it to the model."""
    terms = [term.casefold() for term in _TERM_RE.findall(query) if term.casefold() not in _GENERIC_TERMS]
    if not terms:
        terms = [query.casefold()]
    title = str(row.get("title") or "").casefold()
    body = str(row.get("body") or "").casefold()
    url = str(row.get("href") or "")
    if any(cue in title or cue in body for cue in _SPAM_CUES):
        return 0
    score = sum(3 for term in terms if term in title)
    score += sum(1 for term in terms if term in body)
    if score == 0:
        return 0
    has_meme_cue = any(cue in title or cue in body for cue in _MEME_CUES)
    host = (urlparse(url).hostname or "").casefold()
    if not has_meme_cue and not any(host == domain or host.endswith(f".{domain}") for domain in _SOCIAL_HOSTS):
        return 0
    if has_meme_cue:
        score += 1
    return score


class GetHotTrendsTool(Tool):
    def __init__(self, store: MemeStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "get_hot_trends"

    @property
    def description(self) -> str:
        return (
            "查看微博、B站、抖音、小红书等平台的近期实时热榜候选。"
            "热榜不等于梗；需要理解或使用某个词时，继续调用 search_meme 核实。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "description": "可选平台代码，如 weibo、bilibili"},
                "limit": {"type": "integer", "description": "返回数量，默认 10，最多 30"},
            },
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        limit = min(max(int(kwargs.get("limit", 10)), 1), 30)
        platform = str(kwargs.get("platform") or "").strip() or None
        return _format_trends(self._store.top(limit, platform=platform))


class SearchMemeTool(Tool):
    def __init__(self, store: MemeStore, web_search: SearchFunction | None = None) -> None:
        self._store = store
        self._web_search = web_search or _ddg_search

    @property
    def name(self) -> str:
        return "search_meme"

    @property
    def description(self) -> str:
        return (
            "核实中文网络梗、meme、缩写或流行语的含义、来源和近期用法。"
            "会同时查询本地实时热榜和多个中文网页搜索词；一次只查询一个核心词或短语。"
            "不认识或不确定用法时必须先调用。用户询问来源、出处或原视频时，"
            "最终回复必须附上本工具返回的具体 URL，不能只说网站名称。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要核实的梗、流行语或话题"},
            },
            "required": ["query"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        raw_query = str(kwargs.get("query") or "").strip()[:120]
        if not raw_query:
            return "请提供要查询的梗。"
        query = _normalize_query(raw_query)

        search_queries = [f"{query} meme", f"{query} 是什么梗", f"{query} 表情包 出处"]
        batches = await asyncio.gather(
            *(self._web_search(search_query, 5) for search_query in search_queries),
            return_exceptions=True,
        )
        seen: set[str] = set()
        scored_rows: list[tuple[int, dict[str, str]]] = []
        for batch in batches:
            if isinstance(batch, BaseException):
                continue
            for row in batch:
                url = str(row.get("href") or "")
                key = url or str(row.get("title") or "").casefold()
                if not key or key in seen:
                    continue
                seen.add(key)
                score = _relevance_score(query, row)
                if score:
                    scored_rows.append((score, row))

        scored_rows.sort(key=lambda pair: pair[0], reverse=True)
        web_rows = [row for _, row in scored_rows[:8]]

        sections = [
            f"【查询词】{query}\n"
            f"【本地实时热点匹配】\n{_format_trends(self._store.search(query))}"
        ]
        if web_rows:
            lines = []
            for index, row in enumerate(web_rows, 1):
                lines.append(
                    f"{index}. {row.get('title', '')}\n   {row.get('href', '')}\n   {row.get('body', '')}"
                )
            sections.append(
                "【网页核实结果｜仅为线索，必须交叉判断】\n"
                + "\n\n".join(lines)
                + "\n\n结果可能只是同名内容。若来源互不印证或含义仍不清楚，只能说暂时无法确认，"
                "并请用户提供截图、链接、原句或出处；不能断言这个梗不存在，更不能说用户在瞎编。"
                "如果用户问来源、出处、原视频或哪里来的，最终回复必须原样附上至少一个上述具体 URL；"
                "不要只写平台名，也不要编造或改写链接。"
            )
        else:
            sections.append(
                "【网页核实结果】暂未找到可验证来源。这不代表该梗不存在，它可能很新、只在小圈子/图片/"
                "视频中传播，或搜索引擎尚未收录。请用户提供截图、链接、原句或出处；不要猜测含义，"
                "也绝不能说用户在瞎编。"
            )
        return "\n\n".join(sections)
