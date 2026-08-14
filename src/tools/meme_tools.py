"""Tools for inspecting current trends and verifying meme meanings."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

from src.meme.models import MemeCard
from src.meme.resolver import MemeResolver
from src.meme.store import MemeStore, TrendItem
from src.tools.base import Tool
from src.tools.context import ToolContext
from src.tools.web_search import _ddg_search

_TERM_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]{2,}")
_GENERIC_TERMS = {"什么梗", "是什么梗", "网络用语", "意思", "来源", "梗", "meme"}
_MEME_CUES = ("梗", "meme", "表情包", "网络用语", "什么意思", "来源", "出处", "走红", "流行", "热词")
_SPAM_CUES = ("送彩金", "彩票平台", "棋牌", "投注", "博彩", "官方入口", "app下载")
_SOCIAL_HOSTS = ("bilibili.com", "douyin.com", "xiaohongshu.com", "weibo.com", "zhihu.com")
_QUERY_SUFFIX_RE = re.compile(r"(?:是什么|啥|什么)?(?:梗|meme|表情包|网络用语)+$", re.IGNORECASE)
_ORIGIN_CUES = ("来源", "出处", "原视频", "原帖", "哪里来的", "哪来的")


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


def _format_cards(rows: list[MemeCard]) -> str:
    if not rows:
        return "没有匹配的群内或全局梗卡。"
    lines: list[str] = []
    for card in rows:
        scope = f"群 {card.group_id}" if card.group_id else "全局"
        meaning = card.meaning or "含义仍需结合群内语境"
        lines.append(
            f"- {card.canonical_name} | {scope} | {card.status} | "
            f"置信度 {card.confidence:.2f}\n  含义：{meaning}"
        )
        if card.usage_examples:
            lines.append(f"  群内例句：{card.usage_examples[-1]}")
        if card.source_urls:
            lines.append("  来源：" + " ".join(card.source_urls))
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
    def __init__(self, store: MemeStore) -> None:
        self._store = store
        self._resolver = MemeResolver(store)

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

        resolution = self._resolver.resolve(query, group_id=ctx.group_id)
        local_section = "【本地梗卡｜当前群优先】\n" + _format_cards(resolution.cards)
        trend_section = "【本地实时热点匹配】\n" + _format_trends(resolution.trends)
        asks_origin = any(cue in raw_query for cue in _ORIGIN_CUES)
        has_local_source = bool(resolution.cards and resolution.cards[0].source_urls)
        if resolution.confident and (not asks_origin or has_local_source):
            card = resolution.cards[0]
            source_note = ""
            if card.source_urls:
                source_note = "\n【可核验来源】\n" + "\n".join(card.source_urls)
            return (
                f"【查询词】{query}\n{local_section}\n\n{trend_section}{source_note}\n\n"
                "已优先命中当前群/全局已验证梗卡。回答时说明这是群内约定还是公开来源；"
                "若用户追问出处但卡片没有 URL，仍需请其提供原帖或继续网页核实，不能编造链接。"
            )

        search_queries = [f"{query} meme", f"{query} 是什么梗", f"{query} 表情包 出处"]
        batches = await asyncio.gather(
            *(_ddg_search(search_query, 5) for search_query in search_queries),
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

        sections = [f"【查询词】{query}\n{local_section}\n\n{trend_section}"]
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


class TeachMemeTool(Tool):
    """Allow a current group member to explicitly explain or correct local usage."""

    def __init__(self, store: MemeStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "teach_meme"

    @property
    def description(self) -> str:
        return (
            "记录当前群对某个梗的明确解释、别名、来源或纠错。仅当群成员明确教学或纠正时调用；"
            "不要根据自己的猜测写入。记录默认只在当前群生效。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "梗的名称或核心短语"},
                "meaning": {"type": "string", "description": "群成员明确给出的含义或用法"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "source_urls": {"type": "array", "items": {"type": "string"}},
                "correction": {"type": "boolean", "description": "是否在纠正旧解释"},
            },
            "required": ["name", "meaning"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        if not ctx.group_id:
            return "梗卡按群保存，请在群聊中教学。"
        name = str(kwargs.get("name") or "").strip()[:40]
        meaning = str(kwargs.get("meaning") or "").strip()[:500]
        if not name or not meaning:
            return "需要同时提供梗名和明确含义。"
        aliases = [str(value) for value in kwargs.get("aliases", []) if str(value).strip()][:20]
        urls = [
            str(value) for value in kwargs.get("source_urls", [])
            if str(value).startswith(("http://", "https://"))
        ][:12]
        card = self._store.teach(
            name=name,
            meaning=meaning,
            group_id=ctx.group_id,
            speaker=ctx.user_id or "unknown",
            aliases=aliases,
            source_urls=urls,
            correction=bool(kwargs.get("correction", False)),
        )
        if card is None:
            return "写入失败：梗名、含义或当前群信息无效。"
        action = "已纠正" if kwargs.get("correction") else "已记录"
        return f"{action}本群梗卡「{card.canonical_name}」，置信度 {card.confidence:.2f}。"


class RecallGroupMemesTool(Tool):
    def __init__(self, store: MemeStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "recall_group_memes"

    @property
    def description(self) -> str:
        return "查看当前群已经验证的群内梗和用法，用于自然接梗或核对群内语境。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"limit": {"type": "integer"}}}

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        if not ctx.group_id:
            return "当前不在群聊中。"
        limit = min(max(int(kwargs.get("limit", 10)), 1), 30)
        return _format_cards(self._store.cards_for_group(ctx.group_id, limit=limit))
