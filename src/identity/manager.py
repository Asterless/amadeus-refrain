"""Identity 管理器：加载 Markdown 人设配置，按规则匹配当前人设。

Markdown 格式：
    ## identity_id
    - name: 人设名称
    - priority: 10
    - keywords: 关键词1, 关键词2
    - groups: 群号1, 群号2
    - proactive: （可选）主动插话判断规则

    人设描述正文...
"""

import re
from pathlib import Path

import aiofiles

from src.identity.models import Identity, TriggerRule

_MAX_OVERRIDES = 1000

_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_META_RE = re.compile(r"^-\s+(\w+):\s*(.+)$")


def _parse_markdown(text: str) -> list[Identity]:
    sections = _SECTION_RE.split(text)
    identities: list[Identity] = []
    for i in range(1, len(sections), 2):
        identity_id = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""

        meta: dict[str, str] = {}
        personality_lines: list[str] = []
        past_meta = False

        for line in body.strip().splitlines():
            if not past_meta:
                m = _META_RE.match(line)
                if m:
                    meta[m.group(1)] = m.group(2).strip()
                    continue
                if not line.strip():
                    continue
                past_meta = True
            personality_lines.append(line)

        keywords = [k.strip() for k in meta.get("keywords", "").split(",") if k.strip()]
        groups = [g.strip() for g in meta.get("groups", "").split(",") if g.strip()]
        proactive = meta.get("proactive")

        identities.append(
            Identity(
                id=identity_id,
                name=meta.get("name", identity_id),
                personality="\n".join(personality_lines).strip(),
                trigger=TriggerRule(groups=groups, keywords=keywords),
                priority=int(meta.get("priority", "0")),
                proactive=proactive,
            )
        )
    return identities


class IdentityManager:
    def __init__(self) -> None:
        self._identities: dict[str, Identity] = {}
        self._default: Identity = _builtin_default()
        self._overrides: dict[str, str] = {}

    async def load_file(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        async with aiofiles.open(p, encoding="utf-8") as f:
            text = await f.read()
        for identity in _parse_markdown(text):
            self._identities[identity.id] = identity
            if identity.id == "default":
                self._default = identity

    def resolve(self, session_id: str, group_id: str | None, text: str) -> Identity:
        if session_id in self._overrides:
            override_id = self._overrides[session_id]
            if override_id in self._identities:
                return self._identities[override_id]

        candidates = sorted(self._identities.values(), key=lambda i: i.priority, reverse=True)
        for identity in candidates:
            if identity.id == "default":
                continue
            if _matches(identity, group_id, text):
                return identity

        return self._default

    def switch(self, session_id: str, identity_id: str) -> Identity | None:
        if identity_id not in self._identities:
            return None
        self._overrides[session_id] = identity_id
        if len(self._overrides) > _MAX_OVERRIDES:
            # 移除最早的一半覆盖
            to_remove = list(self._overrides.keys())[: len(self._overrides) // 2]
            for k in to_remove:
                del self._overrides[k]
        return self._identities[identity_id]

    def clear_override(self, session_id: str) -> None:
        self._overrides.pop(session_id, None)

    def list_identities(self) -> list[Identity]:
        return list(self._identities.values())


def _matches(identity: Identity, group_id: str | None, text: str) -> bool:
    rule = identity.trigger
    if rule.groups and (not group_id or group_id not in rule.groups):
        return False
    if rule.keywords:
        return any(kw in text for kw in rule.keywords)
    return bool(rule.groups) and group_id is not None and group_id in rule.groups


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
