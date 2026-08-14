"""Structured models for learned, group-local meme knowledge."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class MemeCard(BaseModel):
    """A bounded piece of meme knowledge with provenance and confidence."""

    id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    meaning: str = ""
    usage_examples: list[str] = Field(default_factory=list)
    usage_scenes: list[str] = Field(default_factory=list)
    avoid_scenes: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    group_id: str | None = None
    confidence: float = Field(default=0.25, ge=0, le=1)
    status: Literal["candidate", "verified", "rejected"] = "candidate"
    evidence_count: int = Field(default=1, ge=0)
    evidence_speakers: list[str] = Field(default_factory=list)
    image_hashes: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)


class MemeObservation(BaseModel):
    """One group-chat observation used to create or update a meme card."""

    group_id: str
    speaker: str
    phrase: str
    text: str
    meaning: str = ""
    context: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    image_hashes: list[str] = Field(default_factory=list)
    explicit: bool = False
    correction: bool = False
    observed_at: datetime = Field(default_factory=utc_now)
