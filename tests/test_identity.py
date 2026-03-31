import pytest

from src.identity.manager import IdentityManager, _parse_markdown


def test_parse_markdown() -> None:
    md = """\
## default
- name: 默认

你是默认人设。

## cat
- name: 猫娘
- priority: 10
- keywords: 喵, 猫娘

你是猫娘。
- 说话带喵
"""
    identities = _parse_markdown(md)
    assert len(identities) == 2

    default = identities[0]
    assert default.id == "default"
    assert default.name == "默认"
    assert "默认人设" in default.personality

    cat = identities[1]
    assert cat.id == "cat"
    assert cat.priority == 10
    assert cat.trigger.keywords == ["喵", "猫娘"]
    assert "猫娘" in cat.personality


def test_parse_markdown_with_groups() -> None:
    md = """\
## vip
- name: VIP
- groups: 111, 222

VIP 人设。
"""
    identities = _parse_markdown(md)
    assert identities[0].trigger.groups == ["111", "222"]


def test_resolve_default() -> None:
    mgr = IdentityManager()
    identity = mgr.resolve("s1", None, "随便说说")
    assert identity.id == "default"


@pytest.fixture
def loaded_mgr(tmp_path: object) -> IdentityManager:
    """同步构建 IdentityManager，直接用 _parse_markdown 填充。"""
    mgr = IdentityManager()
    md = """\
## default
- name: 默认

默认人设。

## cat
- name: 猫娘
- priority: 10
- keywords: 喵, 猫

猫娘人设。

## vip
- name: VIP群
- priority: 5
- groups: 999

VIP 人设。
"""
    for identity in _parse_markdown(md):
        mgr._identities[identity.id] = identity
        if identity.id == "default":
            mgr._default = identity
    return mgr


def test_resolve_by_keyword(loaded_mgr: IdentityManager) -> None:
    identity = loaded_mgr.resolve("s1", "888", "小猫咪你好呀")
    assert identity.id == "cat"


def test_resolve_by_group(loaded_mgr: IdentityManager) -> None:
    identity = loaded_mgr.resolve("s1", "999", "普通消息")
    assert identity.id == "vip"


def test_resolve_no_match(loaded_mgr: IdentityManager) -> None:
    identity = loaded_mgr.resolve("s1", "888", "普通消息")
    assert identity.id == "default"


def test_switch_override(loaded_mgr: IdentityManager) -> None:
    result = loaded_mgr.switch("s1", "cat")
    assert result is not None
    assert result.id == "cat"

    # 覆盖后无论消息内容如何都返回 cat
    identity = loaded_mgr.resolve("s1", None, "普通消息")
    assert identity.id == "cat"


def test_switch_nonexistent(loaded_mgr: IdentityManager) -> None:
    assert loaded_mgr.switch("s1", "nonexistent") is None


def test_clear_override(loaded_mgr: IdentityManager) -> None:
    loaded_mgr.switch("s1", "cat")
    loaded_mgr.clear_override("s1")
    identity = loaded_mgr.resolve("s1", None, "普通消息")
    assert identity.id == "default"


def test_keyword_priority_over_group(loaded_mgr: IdentityManager) -> None:
    """cat(priority=10) > vip(priority=5)，即使在 vip 群里。"""
    identity = loaded_mgr.resolve("s1", "999", "喵喵喵")
    assert identity.id == "cat"
