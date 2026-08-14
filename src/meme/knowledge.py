"""Structured, evidence-backed meme knowledge with lightweight vector retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

_TOKEN_RE = re.compile(r"[0-9A-Za-z]+|[\u4e00-\u9fff]")
_VECTOR_SIZE = 256


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _vectorize(text: str) -> list[float]:
    """Hash character/token features into a normalized local semantic-like vector."""
    values = [0.0] * _VECTOR_SIZE
    normalized = re.sub(r"\s+", "", text.casefold())
    features = list(_TOKEN_RE.findall(normalized))
    features.extend(normalized[index : index + 2] for index in range(max(0, len(normalized) - 1)))
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % _VECTOR_SIZE
        values[index] += 1.0
    length = math.sqrt(sum(value * value for value in values))
    return [value / length for value in values] if length else values


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@dataclass(frozen=True)
class MemeKnowledge:
    canonical: str
    aliases: list[str]
    meaning: str
    origin: str
    usage: str
    examples: list[str]
    evidence: list[str]
    confidence: float
    source_context: str = ""
    updated_at: str = ""


class MemeKnowledgeStore:
    """SQLite-backed verified meme knowledge and vector index."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS meme_knowledge (
                canonical TEXT PRIMARY KEY,
                aliases TEXT NOT NULL,
                meaning TEXT NOT NULL,
                origin TEXT NOT NULL,
                usage TEXT NOT NULL,
                examples TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_context TEXT NOT NULL,
                embedding TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS meme_evidence (
                url TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                observed_at TEXT NOT NULL
            )
            """
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def upsert(
        self,
        *,
        canonical: str,
        aliases: list[str],
        meaning: str,
        origin: str,
        usage: str,
        examples: list[str],
        evidence: list[str],
        confidence: float,
        source_context: str = "",
    ) -> MemeKnowledge:
        canonical = canonical.strip()[:120]
        meaning, origin, usage = (value.strip()[:1000] for value in (meaning, origin, usage))
        aliases = [value.strip()[:80] for value in aliases if value.strip()][:12]
        examples = [value.strip()[:240] for value in examples if value.strip()][:8]
        evidence = [value.strip()[:800] for value in evidence if _is_http_url(value)][:12]
        if not canonical or not meaning or not evidence:
            raise ValueError("canonical、meaning 和至少一个有效 evidence URL 是必填项")
        confidence = max(0.0, min(1.0, float(confidence)))
        updated_at = _now()
        record = MemeKnowledge(
            canonical, aliases, meaning, origin.strip()[:1000], usage,
            examples, evidence, confidence, source_context.strip()[:500], updated_at,
        )
        searchable = " ".join([canonical, *aliases, meaning, record.origin, usage, *examples])
        self._db.execute(
            """
            INSERT INTO meme_knowledge VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical) DO UPDATE SET
              aliases=excluded.aliases, meaning=excluded.meaning, origin=excluded.origin,
              usage=excluded.usage, examples=excluded.examples, evidence=excluded.evidence,
              confidence=excluded.confidence, source_context=excluded.source_context,
              embedding=excluded.embedding, updated_at=excluded.updated_at
            """ ,
            (
                canonical, json.dumps(aliases, ensure_ascii=False), meaning, record.origin,
                usage, json.dumps(examples, ensure_ascii=False), json.dumps(evidence, ensure_ascii=False),
                confidence, record.source_context, json.dumps(_vectorize(searchable)), updated_at,
            ),
        )
        self._db.commit()
        return record

    def record_evidence(self, url: str, title: str = "") -> bool:
        if not _is_http_url(url):
            return False
        self._db.execute(
            "INSERT OR REPLACE INTO meme_evidence(url, title, observed_at) VALUES (?, ?, ?)",
            (url[:800], title[:300], _now()),
        )
        self._db.commit()
        return True

    def has_evidence(self, url: str) -> bool:
        row = self._db.execute("SELECT 1 FROM meme_evidence WHERE url = ?", (url,)).fetchone()
        return row is not None

    def search(self, query: str, limit: int = 8) -> list[MemeKnowledge]:
        query_vector = _vectorize(query)
        rows = self._db.execute("SELECT * FROM meme_knowledge").fetchall()
        scored: list[tuple[float, MemeKnowledge]] = []
        query_folded = query.casefold()
        for row in rows:
            record = _row_to_record(row)
            haystack = " ".join([record.canonical, *record.aliases, record.meaning]).casefold()
            score = _cosine(query_vector, json.loads(row["embedding"]))
            if query_folded in haystack:
                score += 0.75
            score += record.confidence * 0.1
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for score, record in scored[:limit] if score >= 0.15]

    def format_prompt_view(self, limit: int = 8) -> str:
        rows = self.search("近期网络梗 表情包 流行语", limit)
        if not rows:
            return "【已核验梗知识】暂无记录。"
        lines = ["【已核验梗知识｜有证据才可使用】"]
        for record in rows:
            lines.append(f"- {record.canonical}：{record.meaning}（用法：{record.usage or '需结合语境'}）")
        return "\n".join(lines)

    def format_context(self, query: str, limit: int = 4) -> str:
        """Return only records semantically related to the current message."""
        rows = self.search(query, limit)
        if not rows:
            return ""
        lines = ["【当前语境相关的已核验梗｜仅在语气合适时使用】"]
        for record in rows:
            lines.append(
                f"- {record.canonical}：{record.meaning}；用法：{record.usage}；"
                f"证据：{', '.join(record.evidence[:2])}"
            )
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM meme_knowledge").fetchone()[0])


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _row_to_record(row: sqlite3.Row) -> MemeKnowledge:
    return MemeKnowledge(
        canonical=row["canonical"], aliases=json.loads(row["aliases"]), meaning=row["meaning"],
        origin=row["origin"], usage=row["usage"], examples=json.loads(row["examples"]),
        evidence=json.loads(row["evidence"]), confidence=float(row["confidence"]),
        source_context=row["source_context"], updated_at=row["updated_at"],
    )
