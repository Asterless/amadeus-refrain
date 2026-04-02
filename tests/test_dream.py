
import pytest

from src.llm.dream import DreamAgent, dream_pre_check
from src.memory.memo_store import MemoStore


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


def test_pre_check_finds_dangling_refs(store: MemoStore) -> None:
    issues = dream_pre_check(store, user_max_chars=300, group_max_chars=500)
    assert any("999" in issue for issue in issues)


def test_pre_check_no_issues_when_clean(clean_store: MemoStore) -> None:
    issues = dream_pre_check(clean_store, user_max_chars=300, group_max_chars=500)
    assert len(issues) == 0


def test_pre_check_detects_oversized_memo(store: MemoStore) -> None:
    issues = dream_pre_check(store, user_max_chars=10, group_max_chars=10)
    assert any("oversized" in issue for issue in issues)


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


async def test_dream_run_resets_counters(store: MemoStore) -> None:
    agent = DreamAgent(store=store, interval_hours=0, min_compacts=0, max_rounds=5)
    agent._last_dream_time = 0
    agent._compacts_since_dream = 10

    llm_called = False

    async def mock_llm_call(system_prompt: str) -> None:
        nonlocal llm_called
        llm_called = True
        assert "全局索引" in system_prompt or "users" in system_prompt
        assert "999" in system_prompt  # dangling ref should appear

    await agent._run(mock_llm_call)
    assert llm_called
    assert agent._compacts_since_dream == 0
    assert agent._running is False
