import pytest

from src.memory.group_timeline import GroupTimeline
from src.memory.short_term import ShortTermMemory


@pytest.fixture
def short_term() -> ShortTermMemory:
    return ShortTermMemory()


@pytest.fixture
def group_timeline() -> GroupTimeline:
    return GroupTimeline()
