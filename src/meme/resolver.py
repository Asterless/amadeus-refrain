"""Resolve meme queries using scoped knowledge before external search."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.meme.models import MemeCard
from src.meme.store import MemeStore, TrendItem


@dataclass
class MemeResolution:
    query: str
    cards: list[MemeCard] = field(default_factory=list)
    trends: list[TrendItem] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return bool(self.cards and self.cards[0].status == "verified" and self.cards[0].confidence >= 0.75)


class MemeResolver:
    """Deterministic first-stage resolver; web search remains the final fallback."""

    def __init__(self, store: MemeStore) -> None:
        self._store = store

    def resolve(self, query: str, *, group_id: str | None = None, limit: int = 5) -> MemeResolution:
        return MemeResolution(
            query=query,
            cards=self._store.search_cards(query, group_id=group_id, limit=limit),
            trends=self._store.search(query, limit=limit),
        )
