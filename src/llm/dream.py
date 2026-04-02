"""Dream agent: periodic background memory consolidation."""

import asyncio
import time
from collections.abc import Awaitable, Callable

from loguru import logger

from src.memory.memo_store import MemoStore


def dream_pre_check(
    store: MemoStore,
    user_max_chars: int = 300,
    group_max_chars: int = 500,
) -> list[str]:
    """Programmatic scan for structural issues. Returns list of issue descriptions."""
    issues: list[str] = []
    all_ids = set(store.list_ids())
    for mid in all_ids:
        memo = store.read(mid)
        if memo is None:
            continue
        for ref in memo.refs:
            user_ref_id = f"user_{ref}"
            group_ref_id = f"group_{ref}"
            if user_ref_id not in all_ids and group_ref_id not in all_ids:
                issues.append(f"{mid}: reference @{ref} or #{ref} has no corresponding memo")
        if memo.kind == "user" and len(memo.body) > user_max_chars:
            issues.append(f"{mid}: user memo oversized ({len(memo.body)} chars)")
        if memo.kind == "group" and len(memo.body) > group_max_chars:
            issues.append(f"{mid}: group memo oversized ({len(memo.body)} chars)")
    return issues


class DreamAgent:
    def __init__(
        self,
        store: MemoStore,
        interval_hours: int = 24,
        min_compacts: int = 5,
        max_rounds: int = 15,
        user_max_chars: int = 300,
        group_max_chars: int = 500,
    ) -> None:
        self._store = store
        self._interval_hours = interval_hours
        self._min_compacts = min_compacts
        self._max_rounds = max_rounds
        self._user_max_chars = user_max_chars
        self._group_max_chars = group_max_chars
        self._last_dream_time: float = time.time()
        self._compacts_since_dream: int = 0
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None

    def notify_compact(self) -> None:
        """Called after each successful compact."""
        self._compacts_since_dream += 1

    def should_run(self) -> bool:
        if self._running:
            return False
        elapsed_hours = (time.time() - self._last_dream_time) / 3600
        return (
            elapsed_hours >= self._interval_hours
            and self._compacts_since_dream >= self._min_compacts
        )

    async def maybe_run(self, llm_call: Callable[[str], Awaitable[None]]) -> None:
        """Check trigger conditions and run dream if met. Fire-and-forget."""
        if self.should_run():
            self._running = True
            self._task = asyncio.create_task(self._run(llm_call))
            self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            logger.warning("dream task cancelled")
        elif exc := task.exception():
            logger.error("dream task failed: {}", exc)

    async def _run(self, llm_call: Callable[[str], Awaitable[None]]) -> None:
        """Run the dream agent. Should only be called via maybe_run."""
        try:
            logger.info("dream starting")
            issues = dream_pre_check(self._store, self._user_max_chars, self._group_max_chars)
            index_text = self._store.serialize_index()
            issues_text = "\n".join(f"- {i}" for i in issues) if issues else "无明显问题"
            system_prompt = (
                "你是记忆整理助手。以下是当前索引：\n"
                f"{index_text}\n\n"
                f"预检发现以下问题：\n{issues_text}\n\n"
                "请自主检查备忘录质量并修复问题。"
                f"限制：最多 {self._max_rounds} 轮工具调用。"
            )
            await llm_call(system_prompt)
            self._last_dream_time = time.time()
            self._compacts_since_dream = 0
            logger.info("dream completed")
        except Exception:
            logger.exception("dream failed")
        finally:
            self._running = False
