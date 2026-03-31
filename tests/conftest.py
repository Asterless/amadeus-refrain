import pytest

from src.memory.group_context import GroupContext
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory


@pytest.fixture
def short_term() -> ShortTermMemory:
    return ShortTermMemory(max_rounds=5)


@pytest.fixture
def long_term(tmp_path: object) -> LongTermMemory:
    return LongTermMemory(memory_dir=str(tmp_path))


@pytest.fixture
def group_context() -> GroupContext:
    return GroupContext(max_messages=10)
