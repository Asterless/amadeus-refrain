
import json

import pytest

from src.llm.dream import DreamAgent, dream_pre_check
from src.memory.memo_store import MemoStore
from src.sticker.store import StickerStore

# Minimal JPEG bytes for sticker test data
_JPEG_DATA = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"dream-sticker-test"


@pytest.fixture
async def store(tmp_path) -> MemoStore:
    s = MemoStore(base_dir=str(tmp_path))
    await s.startup()
    await s.write("user_100", "用户A｜test\n\n提到 @999(不存在的人)。", "test")
    await s.write("group_200", "群B｜test\n\n@100 活跃。", "test")
    return s


@pytest.fixture
async def clean_store(tmp_path) -> MemoStore:
    s = MemoStore(base_dir=str(tmp_path / "clean"))
    await s.startup()
    await s.write("user_100", "用户A｜test", "test")
    return s


@pytest.fixture
async def pending_store(tmp_path) -> MemoStore:
    s = MemoStore(base_dir=str(tmp_path / "pending"))
    await s.startup()
    await s.write(
        "user_100",
        "@100(测试)\n身份: 学生\n\n## 待整理\n- 喜欢音乐\n- 考试周",
        "test",
    )
    await s.write(
        "group_200",
        "### 成员\n- @100(测试): 学生\n\n## 待整理\n- 讨论了期末考试",
        "test",
    )
    return s


def test_pre_check_finds_dangling_refs(store: MemoStore) -> None:
    issues = dream_pre_check(store, user_max_chars=300, group_max_chars=500)
    assert any("999" in issue for issue in issues)


def test_pre_check_no_issues_when_clean(clean_store: MemoStore) -> None:
    issues = dream_pre_check(clean_store, user_max_chars=300, group_max_chars=500)
    assert len(issues) == 0


def test_pre_check_detects_oversized_memo(store: MemoStore) -> None:
    issues = dream_pre_check(store, user_max_chars=10, group_max_chars=10)
    assert any("oversized" in issue for issue in issues)


def test_pre_check_detects_pending_items(pending_store: MemoStore) -> None:
    issues = dream_pre_check(pending_store, user_max_chars=500, group_max_chars=500)
    pending_issues = [i for i in issues if "待整理" in i]
    assert len(pending_issues) == 2
    assert any("user_100" in i and "2 pending" in i for i in pending_issues)
    assert any("group_200" in i and "1 pending" in i for i in pending_issues)


def test_dream_should_run_when_conditions_met(store: MemoStore) -> None:
    agent = DreamAgent(store=store, interval_hours=0, min_compacts=0, max_rounds=5)
    agent._last_dream_time = 0
    agent._compacts_since_dream = 1
    assert agent.should_run()


def test_dream_should_not_run_too_early(store: MemoStore) -> None:
    agent = DreamAgent(store=store, interval_hours=24, min_compacts=5, max_rounds=5)
    assert not agent.should_run()


def test_dream_should_not_run_insufficient_compacts(store: MemoStore) -> None:
    agent = DreamAgent(store=store, interval_hours=0, min_compacts=5, max_rounds=5)
    agent._last_dream_time = 0
    agent._compacts_since_dream = 3
    assert not agent.should_run()


def test_dream_should_not_run_while_running(store: MemoStore) -> None:
    agent = DreamAgent(store=store, interval_hours=0, min_compacts=0, max_rounds=5)
    agent._last_dream_time = 0
    agent._compacts_since_dream = 10
    agent._running = True
    assert not agent.should_run()


def test_notify_compact_increments(store: MemoStore) -> None:
    agent = DreamAgent(store=store)
    assert agent._compacts_since_dream == 0
    agent.notify_compact()
    agent.notify_compact()
    assert agent._compacts_since_dream == 2


class _FakeToolUse:
    """Minimal stand-in for client._ToolUse to avoid importing private class."""

    def __init__(self, id: str, name: str, input: dict) -> None:
        self.id = id
        self.name = name
        self.input = input


async def test_dream_run_reads_and_writes_memos(pending_store: MemoStore) -> None:
    """Dream tool loop: reads pending memos, then writes merged versions."""
    agent = DreamAgent(
        store=pending_store, interval_hours=0, min_compacts=0, max_rounds=15,
    )
    agent._last_dream_time = 0
    agent._compacts_since_dream = 10

    call_count = 0

    async def mock_api_call(
        system: list, messages: list, tools: list | None = None, max_tokens: int = 1024,
    ) -> dict:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # Round 1: LLM reads both memos
            return {
                "text": "",
                "tool_uses": [
                    _FakeToolUse("r1", "recall_memo", {"id": "user_100"}),
                    _FakeToolUse("r2", "recall_memo", {"id": "group_200"}),
                ],
                "input_tokens": 100, "output_tokens": 50,
                "cache_read": 0, "cache_create": 0,
            }
        if call_count == 2:
            # Round 2: LLM writes merged versions
            return {
                "text": "",
                "tool_uses": [
                    _FakeToolUse("w1", "update_memo", {
                        "id": "user_100",
                        "memo": (
                            "@100(测试)\n"
                            "身份: 学生\n"
                            "备注:\n"
                            "- 喜欢音乐\n"
                            "- 考试周\n\n"
                            "## 待整理"
                        ),
                    }),
                    _FakeToolUse("w2", "update_memo", {
                        "id": "group_200",
                        "memo": (
                            "### 成员\n"
                            "- @100(测试): 学生\n\n"
                            "### 事件\n"
                            "- 讨论了期末考试\n\n"
                            "## 待整理"
                        ),
                    }),
                ],
                "input_tokens": 100, "output_tokens": 50,
                "cache_read": 0, "cache_create": 0,
            }
        # Round 3: done
        return {
            "text": "整理完成。",
            "tool_uses": [],
            "input_tokens": 50, "output_tokens": 10,
            "cache_read": 0, "cache_create": 0,
        }

    await agent._run(mock_api_call)

    assert call_count == 3

    # Verify user memo: pending items merged into main body
    user_memo = pending_store.read("user_100")
    assert user_memo is not None
    assert "喜欢音乐" in user_memo.body
    assert "考试周" in user_memo.body
    # Pending section should be empty (just the header)
    after_pending = user_memo.body.split("## 待整理", 1)[1].strip()
    assert after_pending == ""

    # Verify group memo: pending merged into 事件 section
    group_memo = pending_store.read("group_200")
    assert group_memo is not None
    assert "期末考试" in group_memo.body
    assert "### 事件" in group_memo.body
    after_pending = group_memo.body.split("## 待整理", 1)[1].strip()
    assert after_pending == ""

    # Counters reset
    assert agent._compacts_since_dream == 0
    assert agent._running is False


async def test_dream_cross_validates_memos(pending_store: MemoStore) -> None:
    """Dream reads related memos and fixes inconsistencies across files."""
    # Add a group that references a user with outdated info
    await pending_store.write(
        "group_300",
        "### 成员\n- @100(旧昵称): 毕业生\n\n## 待整理",
        "test",
    )
    agent = DreamAgent(
        store=pending_store, interval_hours=0, min_compacts=0, max_rounds=15,
    )
    agent._last_dream_time = 0
    agent._compacts_since_dream = 10

    call_count = 0

    async def mock_api_call(
        system: list, messages: list, tools: list | None = None, max_tokens: int = 1024,
    ) -> dict:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # Read user_100 and group_300 for cross-validation
            return {
                "text": "读取关联备忘录进行交叉验证",
                "tool_uses": [
                    _FakeToolUse("r1", "recall_memo", {"id": "user_100"}),
                    _FakeToolUse("r2", "recall_memo", {"id": "group_300"}),
                ],
                "input_tokens": 100, "output_tokens": 50,
                "cache_read": 0, "cache_create": 0,
            }
        if call_count == 2:
            # Fix group_300: user_100 is 测试 not 旧昵称, and is 学生 not 毕业生
            return {
                "text": "group_300 中 @100 的信息与 user_100 不一致，已修正",
                "tool_uses": [
                    _FakeToolUse("w1", "update_memo", {
                        "id": "group_300",
                        "memo": "### 成员\n- @100(测试): 学生\n\n## 待整理",
                    }),
                ],
                "input_tokens": 100, "output_tokens": 50,
                "cache_read": 0, "cache_create": 0,
            }
        return {
            "text": "交叉验证完成。",
            "tool_uses": [],
            "input_tokens": 50, "output_tokens": 10,
            "cache_read": 0, "cache_create": 0,
        }

    await agent._run(mock_api_call)

    # group_300 should now have corrected info
    group = pending_store.read("group_300")
    assert group is not None
    assert "测试" in group.body
    assert "学生" in group.body
    assert "旧昵称" not in group.body
    assert "毕业生" not in group.body


async def test_dream_run_resets_counters(store: MemoStore) -> None:
    agent = DreamAgent(store=store, interval_hours=0, min_compacts=0, max_rounds=5)
    agent._last_dream_time = 0
    agent._compacts_since_dream = 10

    async def mock_api_call(
        system: list, messages: list, tools: list | None = None, max_tokens: int = 1024,
    ) -> dict:
        # Verify system prompt contains expected content
        prompt = system[0]["text"]
        assert "索引" in prompt or "users" in prompt
        assert "999" in prompt  # dangling ref should appear
        return {
            "text": "无需处理",
            "tool_uses": [],
            "input_tokens": 50, "output_tokens": 10,
            "cache_read": 0, "cache_create": 0,
        }

    await agent._run(mock_api_call)
    assert agent._compacts_since_dream == 0
    assert agent._running is False


async def test_dream_execute_tool_errors(store: MemoStore) -> None:
    """Tool execution handles missing params and unknown tools gracefully."""
    agent = DreamAgent(store=store)
    assert "缺少 id" in await agent._execute_tool("recall_memo", {})
    assert "缺少 id" in await agent._execute_tool("update_memo", {})
    assert "未找到" in await agent._execute_tool("recall_memo", {"id": "user_nonexistent"})
    assert "未知工具" in await agent._execute_tool("bad_tool", {})


@pytest.fixture
def sticker_store(tmp_path) -> StickerStore:
    return StickerStore(storage_dir=str(tmp_path / "stickers"))


async def test_dream_list_stickers(store: MemoStore, sticker_store: StickerStore) -> None:
    """list_stickers tool returns sticker data as JSON."""
    stk_id, _ = sticker_store.add(_JPEG_DATA, "测试表情", "开心时用", source="auto")
    agent = DreamAgent(store=store, sticker_store=sticker_store)

    result = await agent._execute_tool("list_stickers", {})

    parsed = json.loads(result)
    assert stk_id in parsed
    entry = parsed[stk_id]
    assert entry["description"] == "测试表情"
    assert entry["usage_hint"] == "开心时用"


async def test_dream_delete_sticker(store: MemoStore, sticker_store: StickerStore) -> None:
    """delete_sticker tool removes the sticker and confirms deletion."""
    stk_id, _ = sticker_store.add(_JPEG_DATA, "要删除的表情", "临时用", source="auto")
    assert sticker_store.get(stk_id) is not None

    agent = DreamAgent(store=store, sticker_store=sticker_store)
    result = await agent._execute_tool("delete_sticker", {"id": stk_id})

    assert "已删除" in result
    assert stk_id in result
    assert sticker_store.get(stk_id) is None


async def test_dream_delete_sticker_not_found(store: MemoStore, sticker_store: StickerStore) -> None:
    """delete_sticker returns 未找到 for nonexistent sticker."""
    agent = DreamAgent(store=store, sticker_store=sticker_store)
    result = await agent._execute_tool("delete_sticker", {"id": "stk_nonexistent"})

    assert "未找到" in result
