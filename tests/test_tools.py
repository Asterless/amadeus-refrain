"""工具系统测试：注册表、SSRF 校验、鉴权。"""

import pytest

from src.tools.context import ToolContext
from src.tools.datetime_tool import DateTimeTool
from src.tools.group_admin import MuteUserTool
from src.tools.registry import ToolRegistry
from src.tools.web_fetch import _is_safe_url
from src.tools.web_search import (
    HybridWebSearch,
    WebSearchTool,
    _ddg_search_sync,
    _parse_openai_search_response,
    _rank_results,
)

# ── SSRF 校验 ──


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com", True),
        ("https://api.github.com/repos", True),
        ("http://localhost:8080", False),
        ("http://127.0.0.1:3000", False),
        ("http://10.0.0.1/admin", False),
        ("http://192.168.1.1", False),
        ("http://172.16.0.1", False),
        ("http://169.254.169.254/latest/meta-data/", False),
        ("http://napcat:3001", False),
        ("http://host.docker.internal:34567", False),
        ("", False),
        ("not-a-url", False),
    ],
)
def test_is_safe_url(url: str, expected: bool) -> None:
    assert _is_safe_url(url) == expected


# ── ToolRegistry ──


async def test_registry_call() -> None:
    registry = ToolRegistry()
    registry.register(DateTimeTool())
    ctx = ToolContext(user_id="123")

    result = await registry.call("get_datetime", "{}", ctx)
    assert "20" in result  # 包含年份


async def test_registry_unknown_tool() -> None:
    registry = ToolRegistry()
    ctx = ToolContext(user_id="123")
    result = await registry.call("nonexistent", "{}", ctx)
    assert "未知工具" in result


async def test_registry_to_openai_tools() -> None:
    registry = ToolRegistry()
    registry.register(DateTimeTool())
    tools = registry.to_openai_tools()
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_datetime"


async def test_registry_empty() -> None:
    registry = ToolRegistry()
    assert registry.empty
    registry.register(DateTimeTool())
    assert not registry.empty


# ── 群管理鉴权 ──


async def test_mute_requires_superuser() -> None:
    tool = MuteUserTool(superusers={"admin1"})
    ctx = ToolContext(bot=object(), user_id="regular_user", group_id="123")
    result = await tool.execute(ctx, user_id="target", duration=60)
    assert "权限不足" in result


async def test_mute_requires_group() -> None:
    tool = MuteUserTool(superusers={"admin1"})
    ctx = ToolContext(bot=object(), user_id="admin1", group_id=None)
    result = await tool.execute(ctx, user_id="target", duration=60)
    assert "仅在群聊中" in result


# ── DateTimeTool ──


async def test_datetime_tool() -> None:
    tool = DateTimeTool()
    ctx = ToolContext(user_id="123")
    result = await tool.execute(ctx)
    assert "周" in result  # 包含星期
    assert "-" in result  # 日期格式


# ── ToolRegistry 错误处理 ──


async def test_registry_bad_arguments() -> None:
    registry = ToolRegistry()
    registry.register(DateTimeTool())
    ctx = ToolContext(user_id="123")
    result = await registry.call("get_datetime", "not-json", ctx)
    assert "工具执行出错" in result


# ── WebSearchTool ──


async def test_web_search_formats_results(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_results = [
        {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
        {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
    ]
    monkeypatch.setattr(
        "src.tools.web_search._ddg_search_sync",
        lambda q, n: fake_results,
    )
    tool = WebSearchTool()
    ctx = ToolContext(user_id="123")
    result = await tool.execute(ctx, query="test")
    assert "Result 1" in result
    assert "https://example.com/1" in result
    assert "Result 2" in result
    assert "1." in result and "2." in result


async def test_web_search_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.tools.web_search._ddg_search_sync",
        lambda q, n: [],
    )
    tool = WebSearchTool()
    ctx = ToolContext(user_id="123")
    result = await tool.execute(ctx, query="nonexistent gibberish xyz")
    assert "未找到" in result


async def test_web_search_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_error(q: str, n: int) -> list:
        raise RuntimeError("network error")

    monkeypatch.setattr("src.tools.web_search._ddg_search_sync", raise_error)
    tool = WebSearchTool()
    ctx = ToolContext(user_id="123")
    result = await tool.execute(ctx, query="test")
    assert "搜索失败" in result


async def test_web_search_max_results_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int] = {}

    def capture_n(q: str, n: int) -> list:
        captured["n"] = n
        return []

    monkeypatch.setattr("src.tools.web_search._ddg_search_sync", capture_n)
    tool = WebSearchTool()
    ctx = ToolContext(user_id="123")
    await tool.execute(ctx, query="test", max_results=99)
    assert captured["n"] == 10


def test_web_search_aggregates_engines_and_tolerates_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_backend(query: str, max_results: int, backend: str) -> list[dict[str, str]]:
        calls.append(backend)
        if backend == "google":
            raise RuntimeError("provider unavailable")
        return [
            {
                "title": f"丑橘是什么梗 - {backend}",
                "href": f"https://{backend}.example/result",
                "body": "丑橘是近期流行的橘猫表情包。",
            }
        ]

    monkeypatch.setattr("src.tools.web_search._search_backend_sync", fake_backend)
    results = _ddg_search_sync("丑橘是什么梗", 3)

    assert set(calls) == {"bing", "google", "brave", "duckduckgo"}
    assert len(results) == 3
    assert all("丑橘" in result["title"] for result in results)


def test_web_search_falls_back_to_auto_when_named_engines_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_backend(query: str, max_results: int, backend: str) -> list[dict[str, str]]:
        calls.append(backend)
        if backend != "auto":
            return []
        return [
            {
                "title": "回退结果",
                "href": "https://example.com/fallback",
                "body": "有效摘要",
            }
        ]

    monkeypatch.setattr("src.tools.web_search._search_backend_sync", fake_backend)
    results = _ddg_search_sync("测试查询", 5)

    assert calls.count("auto") == 1
    assert results[0]["title"] == "回退结果"


def test_web_search_ranks_chinese_relevance_and_filters_junk() -> None:
    rows = [
        {
            "title": "Unrelated English result",
            "href": "https://example.com/page",
            "body": "Nothing useful here.",
        },
        {
            "title": "丑橘是什么梗",
            "href": "https://www.bilibili.com/video/1?utm_source=test",
            "body": "丑橘是一只橘猫相关的网络梗。",
        },
        {
            "title": "丑橘是什么梗",
            "href": "https://duplicate.example/page",
            "body": "重复标题。",
        },
        {
            "title": "丑橘字体下载",
            "href": "https://spam.example/font.ttf",
            "body": "font",
        },
        {
            "title": "丑橘官方入口",
            "href": "https://spam.example/page",
            "body": "app下载送彩金",
        },
        {
            "title": "丑橘登录",
            "href": "https://example.com/login?next=home",
            "body": "登录页面",
        },
    ]

    results = _rank_results("丑橘是什么梗", rows)

    assert results[0]["href"] == "https://www.bilibili.com/video/1"
    assert len(results) == 2


def test_openai_web_search_parser_uses_only_returned_sources() -> None:
    payload = {
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "sources": [
                        {"title": "抖音原视频", "url": "https://douyin.com/video/1?utm_source=x"}
                    ]
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "丑橘是一只近期走红的猫。正文中伪造 https://fake.example 不应采纳。",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "B站考据",
                                "url": "https://www.bilibili.com/video/2",
                            }
                        ],
                    }
                ],
            },
        ]
    }

    results = _parse_openai_search_response(payload)

    assert [row["href"] for row in results] == [
        "https://douyin.com/video/1",
        "https://www.bilibili.com/video/2",
    ]
    assert all("fake.example" not in row["href"] for row in results)
    assert all("丑橘是一只" in row["body"] for row in results)


async def test_hybrid_search_survives_openai_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ddgs(query: str, max_results: int) -> list[dict[str, str]]:
        return [
            {
                "title": "丑橘是什么梗",
                "href": "https://www.bilibili.com/video/ddgs",
                "body": "DDGS 搜索结果",
            }
        ]

    class BrokenOpenAI:
        async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
            raise RuntimeError("OpenAI unavailable")

        async def close(self) -> None:
            return None

    monkeypatch.setattr("src.tools.web_search._ddg_search", fake_ddgs)
    service = HybridWebSearch(BrokenOpenAI())  # type: ignore[arg-type]

    results = await service.search("丑橘是什么梗", 5)

    assert results[0]["href"] == "https://www.bilibili.com/video/ddgs"
