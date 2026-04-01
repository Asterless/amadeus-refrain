import pytest

from src.identity.manager import IdentityManager, _parse_identity


def test_parse_identity_basic() -> None:
    md = """\
# 测试人设

你是一个测试机器人。
- 说话简洁
"""
    identity = _parse_identity(md)
    assert identity is not None
    assert identity.name == "测试人设"
    assert "测试机器人" in identity.personality
    assert identity.proactive is None


def test_parse_identity_with_proactive() -> None:
    md = """\
# Bot

Bot 人设描述。

## proactive

有人求助时插话。
用 JSON 回答。
"""
    identity = _parse_identity(md)
    assert identity is not None
    assert identity.name == "Bot"
    assert "Bot 人设描述" in identity.personality
    assert "proactive" not in identity.personality
    assert identity.proactive == "有人求助时插话。\n用 JSON 回答。"


def test_parse_identity_no_title() -> None:
    md = "没有标题的内容"
    assert _parse_identity(md) is None


def test_resolve_default() -> None:
    mgr = IdentityManager()
    identity = mgr.resolve()
    assert identity.id == "default"
    assert identity.name == "默认"


@pytest.mark.asyncio
async def test_load_file(tmp_path: object) -> None:
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "identity.md"
    p.write_text("# 红莉栖\n\n天才物理学家。\n")

    mgr = IdentityManager()
    await mgr.load_file(p)
    identity = mgr.resolve()
    assert identity.name == "红莉栖"
    assert "天才物理学家" in identity.personality
