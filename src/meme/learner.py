"""Conservative group-chat learner for emerging memes."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from src.meme.models import MemeObservation
from src.meme.store import MemeStore
from src.memory.group_timeline import TimelineMessage
from src.memory.types import Content

_EXPLAIN_RE = re.compile(
    r"(?P<name>[0-9A-Za-z\u4e00-\u9fff·_-]{2,24}?)(?:这个)?(?:梗)?"
    r"(?:的意思是|意思是|指的是|出处是|来源是|就是)(?P<meaning>[^\n]{2,160})"
)
_CORRECT_RE = re.compile(
    r"(?P<name>[0-9A-Za-z\u4e00-\u9fff·_-]{2,24})(?:不是这个意思|不是[^，,。]{1,40})"
    r"[，,。 ]*(?:应该是|实际是|是指|意思是)(?P<meaning>[^\n]{2,160})"
)
_URL_RE = re.compile(r"https?://[^\s<>]+")
_PLAIN_PHRASE_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff·_-]{2,18}")
_IGNORE_PHRASES = {
    "哈哈", "哈哈哈", "笑死", "确实", "来了", "好的", "行吧", "不知道",
    "什么意思", "什么梗", "怎么了", "为什么", "谢谢", "可以", "不可以",
}


def _content_text(content: Content) -> str:
    if isinstance(content, str):
        return content.strip()
    return " ".join(block["text"] for block in content if block["type"] == "text").strip()


def _image_hashes(content: Content) -> list[str]:
    if isinstance(content, str):
        return []
    hashes: list[str] = []
    for block in content:
        if block["type"] != "image_ref":
            continue
        path = Path(block.get("original_path") or block["path"])
        try:
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        except OSError:
            continue
    return hashes


class MemeLearner:
    """Extract high-signal explanations and repeated short phrases from a group."""

    def __init__(self, store: MemeStore, *, enabled: bool = True, context_limit: int = 4) -> None:
        self._store = store
        self._enabled = enabled
        self._context_limit = max(1, context_limit)

    def observe(
        self,
        *,
        group_id: str,
        speaker: str,
        content: Content,
        recent: list[TimelineMessage] | None = None,
    ) -> bool:
        if not self._enabled:
            return False
        text = _content_text(content)[:500]
        images = _image_hashes(content)
        if not text and not images:
            return False
        context = [
            f"{row.get('speaker') or '未知'}: {_content_text(row['content'])[:160]}"
            for row in (recent or [])[-self._context_limit :]
            if _content_text(row["content"])
        ]
        urls = _URL_RE.findall(text)

        correction = _CORRECT_RE.search(text)
        if correction:
            return self._store.observe(
                MemeObservation(
                    group_id=group_id,
                    speaker=speaker,
                    phrase=correction.group("name"),
                    meaning=correction.group("meaning"),
                    text=text,
                    context=context,
                    source_urls=urls,
                    image_hashes=images,
                    explicit=True,
                    correction=True,
                )
            )

        explanation = _EXPLAIN_RE.search(text)
        if explanation and any(cue in text for cue in ("梗", "意思", "指的是", "出处", "来源")):
            return self._store.observe(
                MemeObservation(
                    group_id=group_id,
                    speaker=speaker,
                    phrase=explanation.group("name"),
                    meaning=explanation.group("meaning"),
                    text=text,
                    context=context,
                    source_urls=urls,
                    image_hashes=images,
                    explicit=True,
                )
            )

        phrase = text.strip(" \t\r\n，。！？!?~～")
        if (
            not _PLAIN_PHRASE_RE.fullmatch(phrase)
            or phrase in _IGNORE_PHRASES
            or any(cue in phrase for cue in ("什么梗", "什么意思", "是什么", "谁知道"))
            or phrase.startswith("@")
        ):
            return False
        return self._store.observe(
            MemeObservation(
                group_id=group_id,
                speaker=speaker,
                phrase=phrase,
                text=text,
                context=context,
                image_hashes=images,
            )
        )
