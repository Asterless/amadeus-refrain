"""LLM-facing meme tool tests."""

import pytest

from src.meme.store import MemeStore, TrendItem
from src.tools.context import ToolContext
from src.tools.meme_tools import GetHotTrendsTool, SearchMemeTool, _normalize_query


def test_normalize_query_removes_attached_meme_suffix() -> None:
    assert _normalize_query("丑橘meme") == "丑橘"
    assert _normalize_query("丑橘 是什么梗") == "丑橘"
    assert _normalize_query("丑橘表情包") == "丑橘"


async def test_get_hot_trends_filters_platform_and_caps_limit(tmp_path) -> None:
    store = MemeStore(str(tmp_path / "memes.json"))
    store.update(
        [
            TrendItem(platform="weibo", title="微博热点", rank=1),
            TrendItem(platform="bilibili", title="B站热点", rank=1),
        ]
    )
    tool = GetHotTrendsTool(store)

    result = await tool.execute(ToolContext(user_id="1"), platform="weibo", limit=99)
    assert "微博热点" in result
    assert "B站热点" not in result


async def test_search_meme_combines_local_and_deduplicated_web_results(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MemeStore(str(tmp_path / "memes.json"))
    store.update([TrendItem(platform="bilibili", title="测试梗", rank=1)])

    async def fake_search(query: str, limit: int) -> list[dict[str, str]]:
        return [
            {"title": "测试梗是什么意思", "href": "https://example.com/meme", "body": "含义和来源"},
            {"title": "重复结果", "href": "https://example.com/meme", "body": "重复"},
            {"title": "Unrelated GitHub project", "href": "https://github.com/example", "body": "nothing useful"},
            {"title": "测试梗送彩金平台", "href": "https://spam.example", "body": "测试梗官方入口"},
        ]

    monkeypatch.setattr("src.tools.meme_tools._ddg_search", fake_search)
    result = await SearchMemeTool(store).execute(ToolContext(user_id="1"), query="测试梗")

    assert "本地实时热点匹配" in result
    assert "测试梗" in result
    assert "测试梗是什么意思" in result
    assert result.count("https://example.com/meme") == 1
    assert "Unrelated GitHub" not in result
    assert "送彩金" not in result
    assert "最终回复必须原样附上至少一个上述具体 URL" in result


async def test_search_meme_normalizes_attached_suffix_before_search(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MemeStore(str(tmp_path / "memes.json"))
    queries: list[str] = []

    async def fake_search(query: str, limit: int) -> list[dict[str, str]]:
        queries.append(query)
        return [
            {
                "title": "盘点全网最火猫咪表情包出处——丑橘",
                "href": "https://www.bilibili.com/video/test",
                "body": "丑橘是一只走红网络的橘猫表情包。",
            }
        ]

    monkeypatch.setattr("src.tools.meme_tools._ddg_search", fake_search)
    result = await SearchMemeTool(store).execute(ToolContext(user_id="1"), query="丑橘meme")

    assert "【查询词】丑橘" in result
    assert "猫咪表情包出处" in result
    assert all("丑橘meme" not in query for query in queries)
    assert all("丑橘" in query for query in queries)


async def test_search_meme_fails_closed_when_web_search_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MemeStore(str(tmp_path / "memes.json"))

    async def broken_search(query: str, limit: int) -> list[dict[str, str]]:
        raise RuntimeError("offline")

    monkeypatch.setattr("src.tools.meme_tools._ddg_search", broken_search)
    result = await SearchMemeTool(store).execute(ToolContext(user_id="1"), query="陌生梗")
    assert "不代表该梗不存在" in result
    assert "绝不能说用户在瞎编" in result
