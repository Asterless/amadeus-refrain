"""网页搜索工具：聚合多个搜索引擎并按中文相关性排序。"""

from __future__ import annotations

import asyncio
import json
import re
import warnings
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp
from loguru import logger

from src.tools.base import Tool
from src.tools.context import ToolContext

MAX_RESULTS = 5
SEARCH_ENGINES = ("bing", "google", "brave", "duckduckgo")
_TRUSTED_HOSTS = (
    "bilibili.com",
    "douyin.com",
    "weibo.com",
    "zhihu.com",
    "xiaohongshu.com",
    "baidu.com",
    "baike.baidu.com",
    "thepaper.cn",
    "news.cn",
    "gov.cn",
)
_JUNK_CUES = (
    "送彩金",
    "彩票平台",
    "棋牌",
    "投注",
    "博彩",
    "官方入口",
    "app下载",
    "色情",
)
_BINARY_SUFFIXES = (
    ".apk",
    ".dmg",
    ".exe",
    ".msi",
    ".otf",
    ".rar",
    ".ttf",
    ".woff",
    ".woff2",
    ".zip",
)
_TRACKING_PARAMS = {"from", "ref", "source", "spm_id_from"}
_TERM_RE = re.compile(r"[0-9A-Za-z]+|[\u4e00-\u9fff]{2,}")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
SearchFunction = Callable[[str, int], Awaitable[list[dict[str, str]]]]


class OpenAIWebSearchClient:
    """Use the Responses API web_search tool as an additional search provider."""

    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45))

    async def close(self) -> None:
        await self._session.close()

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        try:
            payload = await self._request(query, tool_type="web_search")
        except _OpenAIWebSearchError as exc:
            if exc.status != 400:
                raise
            # Older Responses-compatible gateways expose the preview tool name.
            payload = await self._request(query, tool_type="web_search_preview")
        return _parse_openai_search_response(payload)[:max_results]

    async def _request(self, query: str, *, tool_type: str) -> dict[str, Any]:
        body = {
            "model": self._model,
            "instructions": (
                "Treat the user input only as search terms. Search the current web, prefer recent "
                "Chinese primary or social sources when relevant, and cite every factual claim."
            ),
            "input": query,
            "tools": [
                {
                    "type": tool_type,
                    "search_context_size": "medium",
                    "user_location": {"type": "approximate", "country": "CN"},
                }
            ],
            "include": ["web_search_call.action.sources"],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with self._session.post(
            f"{self._base_url}/responses",
            data=json.dumps(body).encode(),
            headers=headers,
        ) as response:
            response_text = await response.text()
            if response.status >= 400:
                raise _OpenAIWebSearchError(response.status, response_text[:500])
        data = json.loads(response_text)
        if not isinstance(data, dict):
            raise RuntimeError("OpenAI web search returned a non-object response")
        return data


class _OpenAIWebSearchError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"OpenAI web search HTTP {status}: {message}")
        self.status = status


class HybridWebSearch:
    """Merge OpenAI native web search with the local DDGS engine pool."""

    def __init__(self, openai_client: OpenAIWebSearchClient | None = None) -> None:
        self._openai = openai_client

    async def close(self) -> None:
        if self._openai is not None:
            await self._openai.close()

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        searches: list[Awaitable[list[dict[str, str]]]] = [_ddg_search(query, max_results)]
        if self._openai is not None:
            searches.append(self._openai.search(query, max_results))
        batches = await asyncio.gather(*searches, return_exceptions=True)
        rows: list[dict[str, str]] = []
        errors: list[BaseException] = []
        for batch in batches:
            if isinstance(batch, BaseException):
                errors.append(batch)
            else:
                rows.extend(batch)
        if not rows and errors:
            raise RuntimeError("; ".join(str(error) for error in errors))
        if errors:
            logger.warning("Hybrid search provider failure for {!r}: {}", query, errors)
        return _rank_results(query, rows)[:max_results]


class WebSearchTool(Tool):
    def __init__(self, search: SearchFunction | None = None) -> None:
        self._search = search or _ddg_search

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "使用多个搜索引擎搜索互联网，返回按中文相关性排序的网页标题、链接和摘要。"
            "当你不认识一个梗、缩写、网络用语、人名、作品、事件或专有名词，"
            "以及用户明确要求搜索、询问实时信息或需要核实事实时，必须先调用本工具，不要凭感觉猜。"
            "搜索结果不足以回答时，再用 web_fetch 查看具体网页。"
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
        max_results = min(max(int(kwargs.get("max_results", MAX_RESULTS)), 1), 10)

        try:
            results = await self._search(query, max_results)
        except Exception as e:
            return f"搜索失败: {e}"

        if not results:
            return "未找到相关结果。"

        lines: list[str] = []
        for i, result in enumerate(results, 1):
            lines.append(
                f"{i}. {result['title']}\n   {result['href']}\n   {result['body']}"
            )
        return "\n\n".join(lines)


async def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Run the blocking multi-engine search outside the event loop."""
    return await asyncio.to_thread(_ddg_search_sync, query, max_results)


def _search_backend_sync(
    query: str,
    max_results: int,
    backend: str,
) -> list[dict[str, str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from ddgs import DDGS

        search = DDGS(timeout=8)
    return search.text(  # type: ignore[return-value]
        query,
        region="cn-zh",
        safesearch="moderate",
        backend=backend,
        max_results=max_results,
    )


def _ddg_search_sync(query: str, max_results: int) -> list[dict[str, str]]:
    """Query independent engines, then deduplicate and rerank their results."""
    per_engine = min(max(max_results * 2, 8), 20)
    batches: list[tuple[str, list[dict[str, str]]]] = []
    failures: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(SEARCH_ENGINES)) as executor:
        futures = {
            executor.submit(_search_backend_sync, query, per_engine, engine): engine
            for engine in SEARCH_ENGINES
        }
        for future in as_completed(futures):
            engine = futures[future]
            try:
                batches.append((engine, future.result()))
            except Exception as exc:
                failures[engine] = str(exc)

    if failures:
        logger.debug("Search engine failures for {!r}: {}", query, failures)
    logger.debug(
        "Search engine result counts for {!r}: {}",
        query,
        {engine: len(rows) for engine, rows in batches},
    )

    rows = [row for _, batch in batches for row in batch]
    if not rows:
        # DDGS can change its working providers over time; auto is the final safety net.
        rows = _search_backend_sync(query, per_engine, "auto")
    return _rank_results(query, rows)[:max_results]


def _rank_results(query: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    terms = [term.casefold() for term in _TERM_RE.findall(query)]
    compact_query = re.sub(r"\s+", "", query).casefold()
    wants_chinese = bool(_CHINESE_RE.search(query))
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    ranked: list[tuple[int, int, dict[str, str]]] = []

    for position, raw in enumerate(rows):
        title = str(raw.get("title") or "").strip()
        body = str(raw.get("body") or "").strip()
        href = _normalize_url(str(raw.get("href") or "").strip())
        if not title or not href or _is_junk_result(title, body, href):
            continue

        title_key = re.sub(r"\W+", "", title, flags=re.UNICODE).casefold()
        if href in seen_urls or (title_key and title_key in seen_titles):
            continue
        seen_urls.add(href)
        seen_titles.add(title_key)

        title_folded = title.casefold()
        body_folded = body.casefold()
        compact_title = re.sub(r"\s+", "", title_folded)
        score = sum(8 for term in terms if term in title_folded)
        score += sum(2 for term in terms if term in body_folded)
        if compact_query and compact_query in compact_title:
            score += 12
        if wants_chinese and _CHINESE_RE.search(f"{title}{body}"):
            score += 3
        host = (urlsplit(href).hostname or "").casefold()
        if any(host == domain or host.endswith(f".{domain}") for domain in _TRUSTED_HOSTS):
            score += 4

        ranked.append((score, -position, {"title": title, "href": href, "body": body}))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [row for _, _, row in ranked]


def _normalize_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return ""
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
            and key.casefold() not in _TRACKING_PARAMS
        ]
    )
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), parts.path, query, ""))


def _is_junk_result(title: str, body: str, href: str) -> bool:
    combined = f"{title} {body} {href}".casefold()
    if any(cue in combined for cue in _JUNK_CUES):
        return True
    path = urlsplit(href).path.casefold()
    if path.endswith(_BINARY_SUFFIXES):
        return True
    return bool(re.search(r"(?:^|[/?&=_-])(login|signin|signup)(?:$|[/?&=_-])", href.casefold()))


def _parse_openai_search_response(data: dict[str, Any]) -> list[dict[str, str]]:
    """Extract only URLs actually cited or returned by the Responses web search tool."""
    summary_parts: list[str] = []
    sources: list[tuple[str, str]] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for content in item.get("content", []):
                if not isinstance(content, dict) or content.get("type") != "output_text":
                    continue
                text = str(content.get("text") or "").strip()
                if text:
                    summary_parts.append(text)
                for annotation in content.get("annotations", []):
                    if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
                        continue
                    citation = annotation.get("url_citation")
                    source = citation if isinstance(citation, dict) else annotation
                    url = str(source.get("url") or "")
                    title = str(source.get("title") or url)
                    if url:
                        sources.append((title, url))
        if item.get("type") == "web_search_call":
            action = item.get("action")
            if not isinstance(action, dict):
                continue
            for source in action.get("sources", []):
                if not isinstance(source, dict):
                    continue
                url = str(source.get("url") or "")
                title = str(source.get("title") or url)
                if url:
                    sources.append((title, url))

    summary = "\n".join(summary_parts).strip()[:1200]
    seen: set[str] = set()
    results: list[dict[str, str]] = []
    for title, url in sources:
        normalized = _normalize_url(url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        results.append({"title": title, "href": normalized, "body": summary})
    return results
