"""Tool for saving a meme only after the model has extracted and verified it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.meme.knowledge import MemeKnowledgeStore
from src.tools.base import Tool
from src.tools.context import ToolContext


class SaveMemeKnowledgeTool(Tool):
    def __init__(self, store: MemeKnowledgeStore, on_change: Callable[[], None] | None = None) -> None:
        self._store = store
        self._on_change = on_change

    @property
    def name(self) -> str:
        return "save_meme_knowledge"

    @property
    def description(self) -> str:
        return (
            "保存已经从群友说明或网页结果中抽取、并且有具体 URL 证据支持的网络梗知识。"
            "不要保存只有热榜标题的新闻或未经核验的猜测；meaning 要说明梗义，usage 要说明适用语境。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "canonical": {"type": "string", "description": "梗的标准名称"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "meaning": {"type": "string", "description": "梗义和语用含义"},
                "origin": {"type": "string", "description": "来源、原视频或传播背景"},
                "usage": {"type": "string", "description": "适用场景、语气和禁用场景"},
                "examples": {"type": "array", "items": {"type": "string"}},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "网页搜索结果中的完整 URL，至少一个",
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["canonical", "meaning", "origin", "usage", "evidence", "confidence"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        try:
            evidence = [str(value) for value in kwargs.get("evidence", [])]
            unknown = [url for url in evidence if not self._store.has_evidence(url)]
            if unknown:
                return "梗知识未保存：证据 URL 未出现在本轮网页搜索结果中，请先调用 search_meme。"
            record = self._store.upsert(
                canonical=str(kwargs.get("canonical") or ""),
                aliases=[str(value) for value in kwargs.get("aliases", [])],
                meaning=str(kwargs.get("meaning") or ""),
                origin=str(kwargs.get("origin") or ""),
                usage=str(kwargs.get("usage") or ""),
                examples=[str(value) for value in kwargs.get("examples", [])],
                evidence=evidence,
                confidence=float(kwargs.get("confidence", 0)),
                source_context=f"group={ctx.group_id or ''} user={ctx.user_id or ''}",
            )
        except (TypeError, ValueError) as exc:
            return f"梗知识未保存：{exc}"
        if self._on_change is not None:
            self._on_change()
        return f"已保存并向量化梗知识：{record.canonical}（置信度 {record.confidence:.2f}）"
