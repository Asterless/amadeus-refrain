"""Persistent stores for realtime trends and scoped meme knowledge."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.meme.models import MemeCard, MemeObservation

_SPACE_RE = re.compile(r"\s+")


def _now() -> datetime:
    return datetime.now(UTC)


def _clean_text(value: object, limit: int) -> str:
    text = _SPACE_RE.sub(" ", str(value or "")).strip()
    return text.replace("«", "").replace("»", "")[:limit]


class TrendItem(BaseModel):
    """One item observed on a platform's realtime chart."""

    platform: str
    title: str
    url: str = ""
    rank: int = Field(default=999, ge=1)
    hot_value: str = ""
    first_seen_at: datetime = Field(default_factory=_now)
    last_seen_at: datetime = Field(default_factory=_now)
    sightings: int = Field(default=1, ge=1)

    @property
    def key(self) -> str:
        return f"{self.platform.casefold()}::{self.title.casefold()}"


class MemeStore:
    """Stores trend snapshots and exposes a compact prompt/search view."""

    def __init__(
        self,
        path: str,
        *,
        active_hours: int = 72,
        max_entries: int = 500,
        max_prompt_entries: int = 12,
        cards_path: str | None = None,
        candidate_min_sightings: int = 2,
        verified_confidence: float = 0.75,
        max_group_cards: int = 200,
    ) -> None:
        self._path = Path(path)
        self._active_hours = active_hours
        self._max_entries = max_entries
        self._max_prompt_entries = max_prompt_entries
        self._cards_path = Path(cards_path) if cards_path else self._path.with_name("meme_cards.json")
        self._candidate_min_sightings = max(2, candidate_min_sightings)
        self._verified_confidence = min(max(verified_confidence, 0.5), 1.0)
        self._max_group_cards = max(20, max_group_cards)
        self._items: dict[str, TrendItem] = {}
        self._cards: dict[str, MemeCard] = {}
        self._load()
        self._load_cards()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw: Any = json.loads(self._path.read_text(encoding="utf-8"))
            rows = raw.get("items", []) if isinstance(raw, dict) else []
            for row in rows:
                item = TrendItem.model_validate(row)
                self._items[item.key] = item
        except (OSError, ValueError, TypeError):
            # A corrupt cache must never prevent the bot from starting.
            self._items.clear()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = {
            "updated_at": _now().isoformat(),
            "items": [item.model_dump(mode="json") for item in self._items.values()],
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def _load_cards(self) -> None:
        if not self._cards_path.exists():
            return
        try:
            raw: Any = json.loads(self._cards_path.read_text(encoding="utf-8"))
            rows = raw.get("cards", []) if isinstance(raw, dict) else []
            for row in rows:
                card = MemeCard.model_validate(row)
                self._cards[card.id] = card
        except (OSError, ValueError, TypeError):
            self._cards.clear()

    def _save_cards(self) -> None:
        self._cards_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cards_path.with_suffix(self._cards_path.suffix + ".tmp")
        payload = {
            "updated_at": _now().isoformat(),
            "cards": [card.model_dump(mode="json") for card in self._cards.values()],
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._cards_path)

    @staticmethod
    def _card_id(group_id: str | None, name: str) -> str:
        scope = group_id or "global"
        value = f"{scope}::{name.casefold()}".encode()
        return hashlib.sha256(value).hexdigest()[:16]

    @staticmethod
    def _unique(values: list[str], *, limit: int, text_limit: int) -> list[str]:
        result: list[str] = []
        for raw in values:
            value = _clean_text(raw, text_limit)
            if value and value not in result:
                result.append(value)
        return result[-limit:]

    def observe(self, observation: MemeObservation) -> bool:
        """Merge one group observation and promote only independent repetition."""
        name = _clean_text(observation.phrase, 40)
        group_id = _clean_text(observation.group_id, 64)
        speaker = _clean_text(observation.speaker, 80)
        if not name or not group_id or not speaker:
            return False
        card_id = self._card_id(group_id, name)
        old = self._cards.get(card_id)
        if old is None:
            old = MemeCard(id=card_id, canonical_name=name, group_id=group_id, evidence_count=0)

        speakers = self._unique([*old.evidence_speakers, speaker], limit=30, text_limit=80)
        is_new_speaker = speaker not in old.evidence_speakers
        evidence_count = old.evidence_count + (1 if is_new_speaker else 0)
        meaning = _clean_text(observation.meaning, 500) or old.meaning
        corrections = old.corrections
        if observation.correction:
            if old.meaning and old.meaning != meaning:
                corrections = self._unique(
                    [*old.corrections, f"原解释：{old.meaning}；纠正为：{meaning}"],
                    limit=10,
                    text_limit=600,
                )
            confidence = max(0.75, old.confidence - 0.1)
            status = "verified"
        elif observation.explicit:
            confidence = max(old.confidence, 0.85)
            status = "verified"
        elif len(speakers) >= self._candidate_min_sightings:
            confidence = max(old.confidence, self._verified_confidence)
            status = "verified"
        else:
            confidence = min(0.6, old.confidence + (0.1 if is_new_speaker else 0.02))
            status = old.status

        examples = [*old.usage_examples, *observation.context, observation.text]
        card = old.model_copy(
            update={
                "meaning": meaning,
                "confidence": confidence,
                "status": status,
                "evidence_count": evidence_count,
                "evidence_speakers": speakers,
                "usage_examples": self._unique(examples, limit=8, text_limit=220),
                "source_urls": self._unique(
                    [*old.source_urls, *observation.source_urls], limit=12, text_limit=500
                ),
                "image_hashes": self._unique(
                    [*old.image_hashes, *observation.image_hashes], limit=12, text_limit=64
                ),
                "corrections": corrections,
                "last_seen_at": observation.observed_at,
            }
        )
        self._cards[card_id] = card
        self._prune_cards(group_id)
        self._save_cards()
        return True

    def teach(
        self,
        *,
        name: str,
        meaning: str,
        group_id: str,
        speaker: str,
        aliases: list[str] | None = None,
        source_urls: list[str] | None = None,
        correction: bool = False,
    ) -> MemeCard | None:
        """Explicit tool-facing teaching API."""
        observation = MemeObservation(
            group_id=group_id,
            speaker=speaker,
            phrase=name,
            text=f"{name}：{meaning}",
            meaning=meaning,
            source_urls=source_urls or [],
            explicit=True,
            correction=correction,
        )
        if not self.observe(observation):
            return None
        card = self._cards[self._card_id(group_id, _clean_text(name, 40))]
        if aliases:
            card.aliases = self._unique([*card.aliases, *aliases], limit=20, text_limit=40)
            self._save_cards()
        return card

    def _prune_cards(self, group_id: str) -> None:
        rows = [card for card in self._cards.values() if card.group_id == group_id]
        if len(rows) <= self._max_group_cards:
            return
        rows.sort(key=lambda card: (card.status == "verified", card.confidence, card.last_seen_at), reverse=True)
        keep = {card.id for card in rows[: self._max_group_cards]}
        self._cards = {
            key: card for key, card in self._cards.items()
            if card.group_id != group_id or key in keep
        }

    def search_cards(
        self, query: str, *, group_id: str | None = None, limit: int = 8
    ) -> list[MemeCard]:
        needle = _clean_text(query, 120).casefold()
        if not needle:
            return []
        scored: list[tuple[float, MemeCard]] = []
        for card in self._cards.values():
            if card.status == "rejected" or card.group_id not in (None, group_id):
                continue
            names = [card.canonical_name, *card.aliases]
            similarity = max(SequenceMatcher(None, needle, name.casefold()).ratio() for name in names)
            exact = any(needle == name.casefold() for name in names)
            contains = any(needle in name.casefold() or name.casefold() in needle for name in names)
            if not exact and not contains and similarity < 0.62:
                continue
            scope = 1.0 if group_id and card.group_id == group_id else 0.0
            score = (3.0 if exact else 1.5 if contains else similarity) + scope + card.confidence
            scored.append((score, card))
        scored.sort(key=lambda row: (row[0], row[1].last_seen_at), reverse=True)
        return [card for _, card in scored[: max(0, limit)]]

    def cards_for_group(self, group_id: str, *, limit: int = 20) -> list[MemeCard]:
        rows = [
            card for card in self._cards.values()
            if card.status == "verified" and card.group_id in (None, group_id)
        ]
        rows.sort(
            key=lambda card: (card.group_id == group_id, card.confidence, card.last_seen_at),
            reverse=True,
        )
        return rows[: max(0, limit)]

    def update(self, incoming: list[TrendItem], *, now: datetime | None = None) -> bool:
        """Merge a snapshot, prune stale rows, and persist if anything changed."""
        observed_at = now or _now()
        changed = False
        for raw in incoming:
            platform = _clean_text(raw.platform, 32).casefold()
            title = _clean_text(raw.title, 160)
            if not platform or not title:
                continue
            item = raw.model_copy(
                update={
                    "platform": platform,
                    "title": title,
                    "url": _clean_text(raw.url, 500),
                    "hot_value": _clean_text(raw.hot_value, 80),
                    "last_seen_at": observed_at,
                }
            )
            old = self._items.get(item.key)
            if old is not None:
                item.first_seen_at = old.first_seen_at
                item.sightings = old.sightings + 1
            self._items[item.key] = item
            changed = True

        cutoff = observed_at - timedelta(hours=self._active_hours)
        stale = [key for key, item in self._items.items() if item.last_seen_at < cutoff]
        for key in stale:
            del self._items[key]
            changed = True

        if len(self._items) > self._max_entries:
            keep = self.top(self._max_entries, now=observed_at, include_stale=True)
            self._items = {item.key: item for item in keep}
            changed = True

        if changed:
            self._save()
        return changed

    @staticmethod
    def _score(item: TrendItem, now: datetime) -> float:
        age_hours = max(0.0, (now - item.last_seen_at).total_seconds() / 3600)
        recency = math.exp(-age_hours / 24)
        rank_score = 1 / max(1, item.rank)
        repeat_score = min(item.sightings, 12) / 12
        return recency * 0.55 + rank_score * 0.35 + repeat_score * 0.10

    def top(
        self,
        limit: int,
        *,
        platform: str | None = None,
        now: datetime | None = None,
        include_stale: bool = False,
    ) -> list[TrendItem]:
        current = now or _now()
        cutoff = current - timedelta(hours=self._active_hours)
        wanted = platform.casefold() if platform else None
        rows = [
            item
            for item in self._items.values()
            if (include_stale or item.last_seen_at >= cutoff)
            and (wanted is None or item.platform == wanted)
        ]
        rows.sort(key=lambda item: self._score(item, current), reverse=True)
        return rows[: max(0, limit)]

    def search(self, query: str, limit: int = 8) -> list[TrendItem]:
        needle = _clean_text(query, 120).casefold()
        if not needle:
            return []
        tokens = [token for token in re.split(r"\s+", needle) if token]
        rows = [
            item
            for item in self.top(self._max_entries)
            if needle in item.title.casefold()
            or item.title.casefold() in needle
            or all(token in item.title.casefold() for token in tokens)
        ]
        return rows[:limit]

    def format_prompt_view(self, group_id: str | None = None) -> str:
        rows = self.top(self._max_prompt_entries)
        cards = self.cards_for_group(group_id, limit=self._max_prompt_entries) if group_id else []
        lines: list[str] = []
        if cards:
            lines.extend([
                "【当前群已验证梗卡】",
                "这些是本群观察到的用法；仅在语境自然匹配时使用，不要生硬复读。",
            ])
            for card in cards:
                meaning = f"：{card.meaning}" if card.meaning else "（含义仍依赖群内语境）"
                lines.append(f"- {card.canonical_name}{meaning}")
        if lines:
            lines.append("")
        if not rows:
            lines.append("【实时热点候选】暂无可用数据。")
            return "\n".join(lines)
        lines.extend([
            "【实时热点候选｜不可信外部数据】",
            "以下只是近期热榜标题，不代表它们是梗，也不代表适合发言。",
            "只有当前对话自然相关时才考虑；不理解含义必须先调用 search_meme 核实，严禁拿严肃新闻硬玩梗。",
        ])
        for item in rows:
            lines.append(f"- [{item.platform}] #{item.rank} {item.title}")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def card_count(self) -> int:
        return len(self._cards)
