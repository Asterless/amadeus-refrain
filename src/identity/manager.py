"""Identity 管理器：加载单人格 Markdown 配置。

Markdown 格式：
    # 人设名称

    人设描述正文...

    ## proactive
    （可选）主动插话判断规则，支持多行。
"""

import re
from pathlib import Path

import aiofiles

from src.identity.models import Identity

_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_PROACTIVE_RE = re.compile(r"^##\s+proactive\s*$", re.MULTILINE | re.IGNORECASE)


def _parse_identity(text: str) -> Identity | None:
    """解析单人格 Markdown 文件。"""
    m = _TITLE_RE.search(text)
    if not m:
        return None

    name = m.group(1).strip()
    body = text[m.end() :].strip()

    # 分割 personality 和 proactive
    proactive: str | None = None
    split = _PROACTIVE_RE.search(body)
    if split:
        personality = body[: split.start()].strip()
        proactive = body[split.end() :].strip() or None
    else:
        personality = body

    return Identity(
        id="default",
        name=name,
        personality=personality,
        proactive=proactive,
    )


class IdentityManager:
    def __init__(self) -> None:
        self._identity: Identity = _builtin_default()

    async def load_file(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        async with aiofiles.open(p, encoding="utf-8") as f:
            text = await f.read()
        identity = _parse_identity(text)
        if identity:
            self._identity = identity

    def resolve(self) -> Identity:
        return self._identity


def _builtin_default() -> Identity:
    return Identity(
        id="default",
        name="默认",
        personality=(
            "你是一个QQ群聊机器人。\n"
            "- 说话简洁自然，像真人群友\n"
            "- 适当使用网络用语和表情\n"
            "- 不要自称\"AI\"或\"语言模型\"\n"
            "- 回复不要太长，一般1-3句话"
        ),
    )
