import pytest

from src.memory.memo_store import MemoStore, parse_memo

SAMPLE_USER_MEMO = """\
<!-- updated: 2026-03-31 14:30 | source: compact:group:987654 -->

小明（明哥）｜杭州·后端·Go/Python

最近在学 Rust。和 @789012(小红) 在 #987654 互怼。
"""

SAMPLE_GROUP_MEMO = """\
<!-- updated: 2026-03-31 14:30 | source: compact:group:987654 -->

技术吹水群｜~10人｜轻松玩梗

@123456(小明) 和 @789012(小红) 经常互怼前后端。
"""


def test_parse_user_memo() -> None:
    memo = parse_memo("user_123456", SAMPLE_USER_MEMO)
    assert memo.id == "user_123456"
    assert memo.kind == "user"
    assert memo.identity == "小明（明哥）｜杭州·后端·Go/Python"
    assert memo.source == "compact:group:987654"
    assert "789012" in memo.refs
    assert "987654" in memo.refs


def test_parse_group_memo() -> None:
    memo = parse_memo("group_987654", SAMPLE_GROUP_MEMO)
    assert memo.kind == "group"
    assert "技术吹水群" in memo.identity


@pytest.fixture
def store(tmp_path) -> MemoStore:
    return MemoStore(base_dir=str(tmp_path))


def test_store_read_before_startup_raises(store: MemoStore) -> None:
    with pytest.raises(RuntimeError, match="startup"):
        store.read("user_999")


async def test_store_empty_read(store: MemoStore) -> None:
    await store.startup()
    assert store.read("user_999") is None


async def test_store_write_and_read(store: MemoStore) -> None:
    await store.startup()
    await store.write("user_123456", "小明（明哥）｜杭州·后端·Go/Python\n\n最近在学 Rust。", "test")
    memo = store.read("user_123456")
    assert memo is not None
    assert memo.kind == "user"
    assert "小明" in memo.identity


async def test_store_mentions_index(store: MemoStore) -> None:
    await store.startup()
    await store.write("group_987654", "技术群｜5人\n\n@123456(小明) 和 @789012 活跃。", "test")
    mentioned_in = store.about("123456")
    assert any(m.id == "group_987654" for m in mentioned_in)


async def test_store_list_ids(store: MemoStore) -> None:
    await store.startup()
    await store.write("user_111", "测试｜test", "test")
    await store.write("group_222", "群｜test", "test")
    assert "user_111" in store.list_ids("user")
    assert "group_222" in store.list_ids("group")
    all_ids = store.list_ids()
    assert "user_111" in all_ids
    assert "group_222" in all_ids


async def test_store_serialize_index(store: MemoStore) -> None:
    await store.startup()
    await store.write("user_123456", "小明｜杭州后端\n\n和 @789012 互怼。在 #987654 活跃。", "test")
    index = store.serialize_index()
    assert "@123456" in index
    assert "小明" in index


async def test_write_creates_dirs(store: MemoStore) -> None:
    """Write to a new user should create the users/ directory."""
    await store.startup()
    await store.write("user_999", "新用户｜test", "test")
    assert store.read("user_999") is not None


async def test_path_traversal_rejected(store: MemoStore) -> None:
    """IDs with .. or / should be rejected."""
    await store.startup()
    with pytest.raises(ValueError):
        await store.write("user_../../etc/passwd", "hacker", "test")


async def test_atomic_write_no_tmp_residual(store: MemoStore) -> None:
    """After write, no .tmp files should remain."""
    await store.startup()
    await store.write("user_100", "Test｜test", "test")
    tmp_files = list(store._base_dir.rglob("*.tmp"))
    assert len(tmp_files) == 0


async def test_changelog_appended(store: MemoStore) -> None:
    """Each write appends one line to .log."""
    await store.startup()
    await store.write("user_100", "V1｜first", "compact:group:999")
    await store.write("user_100", "V2｜second", "tool:abc")
    log_path = store._base_dir / "users" / "100.log"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert "compact:group:999" in lines[0]
    assert "tool:abc" in lines[1]


async def test_startup_cleans_tmp(store: MemoStore, tmp_path) -> None:
    """Startup removes leftover .tmp files."""
    users_dir = store._base_dir / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    (users_dir / "orphan.md.tmp").write_text("garbage")
    await store.startup()
    assert not (users_dir / "orphan.md.tmp").exists()


async def test_concurrent_writes_different_ids(store: MemoStore) -> None:
    """Writes to different IDs run in parallel without error."""
    import asyncio

    await store.startup()
    await asyncio.gather(
        store.write("user_1", "A｜a", "test"),
        store.write("user_2", "B｜b", "test"),
        store.write("group_3", "C｜c", "test"),
    )
    assert store.read("user_1") is not None
    assert store.read("user_2") is not None
    assert store.read("group_3") is not None


async def test_full_lifecycle(tmp_path) -> None:
    """Startup → write users + group → read → mentions → index → overwrite → verify persistence."""
    store = MemoStore(base_dir=str(tmp_path))
    await store.startup()

    # Create
    await store.write("user_100", "小明｜杭州后端\n\n和 @200(小红) 在 #300 互怼。", "compact:group:300")
    await store.write("user_200", "小红｜前端\n\n和 @100(小明) 互怼。", "compact:group:300")
    await store.write("group_300", "技术群｜5人\n\n@100 @200 活跃。", "compact:group:300")

    # Read
    memo = store.read("user_100")
    assert memo is not None
    assert memo.identity == "小明｜杭州后端"

    # Mentions
    mentioned_in = store.about("100")
    ids = {m.id for m in mentioned_in}
    assert "user_200" in ids      # 小红's memo mentions @100
    assert "group_300" in ids     # group memo mentions @100

    # Index
    index = store.serialize_index()
    assert "@100" in index
    assert "@200" in index
    assert "#300" in index

    # Overwrite (full rewrite)
    await store.write("user_100", "小明（老明）｜杭州后端·转Go\n\n换了工作。", "tool:sess1")
    memo = store.read("user_100")
    assert memo is not None
    assert "老明" in memo.identity
    assert "转Go" in memo.identity

    # Old mentions should be cleaned up after overwrite
    mentioned_in_after = store.about("200")
    ids_after = {m.id for m in mentioned_in_after}
    # user_100 no longer mentions @200 after overwrite
    assert "user_100" not in ids_after

    # Changelog
    log_path = tmp_path / "users" / "100.log"
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2

    # Restart and verify persistence
    store2 = MemoStore(base_dir=str(tmp_path))
    await store2.startup()
    reloaded = store2.read("user_100")
    assert reloaded is not None
    assert "老明" in reloaded.identity
    assert len(store2.list_ids("user")) == 2
    assert len(store2.list_ids("group")) == 1

    # Mentions survive restart
    mentioned_in_restart = store2.about("100")
    restart_ids = {m.id for m in mentioned_in_restart}
    assert "group_300" in restart_ids


# ---------------------------------------------------------------------------
# Append — pending section
# ---------------------------------------------------------------------------


async def test_append_creates_pending_section(store: MemoStore) -> None:
    """append() on a memo without 待整理 creates the section."""
    await store.startup()
    await store.write("user_100", "@100(测试)\n身份: 学生", "test")
    await store.append("user_100", "喜欢打篮球", "compact:private:user_100")
    memo = store.read("user_100")
    assert memo is not None
    assert "## 待整理" in memo.body
    assert "- 喜欢打篮球" in memo.body


async def test_append_adds_to_existing_pending(store: MemoStore) -> None:
    """Second append() adds a new bullet to the existing 待整理 section."""
    await store.startup()
    await store.write("user_100", "@100(测试)\n\n## 待整理\n- 第一条", "test")
    await store.append("user_100", "第二条", "compact:private:user_100")
    memo = store.read("user_100")
    assert memo is not None
    assert "- 第一条" in memo.body
    assert "- 第二条" in memo.body


async def test_append_new_memo_creates_pending(store: MemoStore) -> None:
    """append() on a non-existent memo creates it with just a 待整理 section."""
    await store.startup()
    await store.append("user_200", "新用户观察", "compact:private:user_200")
    memo = store.read("user_200")
    assert memo is not None
    assert "## 待整理" in memo.body
    assert "- 新用户观察" in memo.body
