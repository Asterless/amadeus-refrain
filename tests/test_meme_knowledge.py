"""Structured meme knowledge and vector retrieval tests."""

from src.meme.knowledge import MemeKnowledgeStore
from src.meme.store import MemeStore
from src.tools.context import ToolContext
from src.tools.meme_learning import SaveMemeKnowledgeTool
from src.tools.meme_tools import SearchMemeTool


def test_knowledge_store_requires_evidence_and_persists_vectors(tmp_path) -> None:
    path = tmp_path / "meme_knowledge.db"
    store = MemeKnowledgeStore(str(path))
    store.record_evidence("https://www.douyin.com/video/123", "丑橘来源")
    record = store.upsert(
        canonical="丑橘",
        aliases=["丑橘猫"],
        meaning="一只因贴脸哭泣视频走红的橘猫，也可指相关反应图",
        origin="抖音视频传播",
        usage="表达委屈、崩溃或无语时使用",
        examples=["我现在就是丑橘"],
        evidence=["https://www.douyin.com/video/123"],
        confidence=0.9,
    )
    assert record.canonical == "丑橘"
    assert store.count == 1
    assert store.search("那只哭泣的橘猫")[0].canonical == "丑橘"
    assert "丑橘" in store.format_context("哭泣的橘猫")
    store.close()

    reopened = MemeKnowledgeStore(str(path))
    assert reopened.search("丑橘猫")[0].evidence == ["https://www.douyin.com/video/123"]
    reopened.close()


def test_save_tool_rejects_missing_or_invalid_evidence(tmp_path) -> None:
    store = MemeKnowledgeStore(str(tmp_path / "knowledge.db"))
    tool = SaveMemeKnowledgeTool(store)

    result = __import__("asyncio").run(
        tool.execute(
            ToolContext(group_id="1", user_id="2"),
            canonical="未知梗",
            meaning="猜测",
            origin="不清楚",
            usage="不清楚",
            evidence=["不是链接"],
            confidence=0.2,
        )
    )

    assert "未保存" in result
    assert store.count == 0
    store.close()


def test_save_tool_invalidates_prompt_cache_after_success(tmp_path) -> None:
    store = MemeKnowledgeStore(str(tmp_path / "knowledge.db"))
    store.record_evidence("https://example.com/source", "来源")
    changed: list[bool] = []
    tool = SaveMemeKnowledgeTool(store, on_change=lambda: changed.append(True))

    import asyncio

    result = asyncio.run(
        tool.execute(
            ToolContext(group_id="1", user_id="2"),
            canonical="新梗",
            meaning="表达突然沉默",
            origin="群聊传播",
            usage="面对离谱发言时",
            evidence=["https://example.com/source"],
            confidence=0.8,
        )
    )

    assert "已保存" in result
    assert changed == [True]
    store.close()


async def test_group_explanation_search_and_save_form_verified_chain(tmp_path, monkeypatch) -> None:
    trends = MemeStore(str(tmp_path / "trends.json"))
    knowledge = MemeKnowledgeStore(str(tmp_path / "knowledge.db"))

    async def fake_search(query: str, limit: int) -> list[dict[str, str]]:
        return [
            {
                "title": "丑橘猫原视频出处",
                "href": "https://www.douyin.com/video/7616159047347111987",
                "body": "丑橘边嚼口香糖边哭泣的视频走红，后来用于表达委屈。",
            }
        ]

    monkeypatch.setattr("src.tools.meme_tools._ddg_search", fake_search)
    result = await SearchMemeTool(trends, knowledge=knowledge).execute(
        ToolContext(group_id="42", user_id="7"), query="丑橘是什么梗"
    )
    assert "https://www.douyin.com/video/7616159047347111987" in result

    saved = await SaveMemeKnowledgeTool(knowledge).execute(
        ToolContext(group_id="42", user_id="7"),
        canonical="丑橘",
        aliases=["丑橘猫"],
        meaning="因哭泣视频走红的橘猫梗，用来表达委屈或崩溃",
        origin="抖音原视频",
        usage="遇到委屈、无语或离谱处境时使用",
        examples=["我现在就是丑橘"],
        evidence=["https://www.douyin.com/video/7616159047347111987"],
        confidence=0.9,
    )
    assert "已保存" in saved
    assert knowledge.search("哭泣橘猫")[0].canonical == "丑橘"
    knowledge.close()
