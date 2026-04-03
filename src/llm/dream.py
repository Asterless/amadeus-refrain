"""Dream agent: periodic background memory consolidation with LLM tool loop."""

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from loguru import logger

from src.memory.memo_store import GROUP_MEMO_TEMPLATE, PENDING_HEADER, USER_MEMO_TEMPLATE, MemoStore

# Type alias for the LLM API caller (matches LLMClient._call signature)
ApiCaller = Callable[..., Awaitable[dict[str, Any]]]

# Dedicated dream logger — writes only to dream log files, not the main bot log.
dream_logger = logger.bind(dream=True)


def setup_dream_logger(log_dir: str) -> None:
    """Add a dedicated sink for dream logs, filtered from the main bot log."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger.add(
        path / "dream_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        level="DEBUG",
        filter=lambda record: record["extra"].get("dream", False),
    )

_RECALL_MEMO_TOOL: dict[str, Any] = {
    "name": "recall_memo",
    "description": "读取指定备忘录的完整内容。",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "memo ID，如 user_123456 或 group_987654",
            },
        },
        "required": ["id"],
    },
}

_UPDATE_MEMO_TOOL: dict[str, Any] = {
    "name": "update_memo",
    "description": "完整覆写指定备忘录。写入的内容会替换原有全部内容。",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "memo ID，如 user_123456 或 group_987654",
            },
            "memo": {
                "type": "string",
                "description": "完整的新内容（全量覆写，不是追加）",
            },
        },
        "required": ["id", "memo"],
    },
}


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
        if PENDING_HEADER in memo.body:
            after_header = memo.body.split(PENDING_HEADER, 1)[1].strip()
            if after_header:
                n = sum(1 for line in after_header.splitlines() if line.strip().startswith("- "))
                if n > 0:
                    issues.append(f"{mid}: has {n} pending item(s) in 待整理 to merge")
    return issues


class DreamAgent:
    def __init__(
        self,
        store: MemoStore,
        interval_hours: int = 24,
        max_rounds: int = 15,
        user_max_chars: int = 300,
        group_max_chars: int = 500,
    ) -> None:
        self._store = store
        self._interval_hours = interval_hours
        self._max_rounds = max_rounds
        self._user_max_chars = user_max_chars
        self._group_max_chars = group_max_chars
        self._running: bool = False
        self._loop_task: asyncio.Task[None] | None = None

    def start(self, api_call: ApiCaller) -> None:
        """Start the independent background dream loop."""
        if self._loop_task is not None:
            return
        self._loop_task = asyncio.create_task(self._loop(api_call))
        self._loop_task.add_done_callback(self._on_loop_done)
        dream_logger.info("dream loop started | interval={}h", self._interval_hours)

    async def stop(self) -> None:
        """Cancel the background loop and wait for it to finish."""
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
            dream_logger.info("dream loop stopped")

    async def _loop(self, api_call: ApiCaller) -> None:
        """Sleep → run → repeat. First run waits a full interval."""
        interval_s = self._interval_hours * 3600
        while True:
            await asyncio.sleep(interval_s)
            await self._run(api_call)

    def _on_loop_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        if exc := task.exception():
            dream_logger.error("dream loop crashed: {}", exc)

    async def _run(self, api_call: ApiCaller) -> None:
        """Run the dream agent with a tool loop for memo consolidation."""
        self._running = True
        t0 = time.time()
        try:
            dream_logger.info("dream starting")
            issues = dream_pre_check(self._store, self._user_max_chars, self._group_max_chars)
            index_text = self._store.serialize_index()
            issues_text = "\n".join(f"- {i}" for i in issues) if issues else "无明显问题"

            template_section = (
                f"\n用户备忘录模板：\n{USER_MEMO_TEMPLATE}\n\n"
                f"群备忘录模板：\n{GROUP_MEMO_TEMPLATE}\n"
            )
            system = [{"type": "text", "text": (
                "你是记忆整理助手。以下是当前备忘录索引：\n"
                f"{index_text}\n\n"
                f"预检发现以下问题：\n{issues_text}\n\n"
                "请依次处理：\n"
                "1. 有「待整理」项的备忘录：用 recall_memo 读取完整内容，"
                "将待整理的条目合并到主体对应位置，"
                "清空「## 待整理」区域（保留空的标题行），用 update_memo 写回。\n"
                "2. 跨文件交叉验证：读取相关联的备忘录（如某用户提到的群、群里提到的用户），"
                "检查信息是否一致，修正矛盾或过时内容。\n"
                "3. 其他问题（悬空引用、超长等）：自主修复。\n"
                f"{template_section}"
                f"字数限制：用户 {self._user_max_chars} 字，群 {self._group_max_chars} 字。\n\n"
                f"限制：最多 {self._max_rounds} 轮工具调用。如果没有问题需要处理，直接回复即可。"
            )}]

            tools = [_RECALL_MEMO_TOOL, _UPDATE_MEMO_TOOL]
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "请开始整理备忘录。"},
            ]

            memo_reads = 0
            memo_writes = 0

            for round_i in range(self._max_rounds):
                result = await api_call(system, messages, tools, 2048)

                text: str = result["text"].strip()
                tool_uses: list[Any] = result["tool_uses"]

                if not tool_uses:
                    dream_logger.info(
                        "dream finished | rounds={} reads={} writes={} elapsed={:.1f}s",
                        round_i + 1, memo_reads, memo_writes, time.time() - t0,
                    )
                    break

                # Build assistant message with text + tool_use blocks
                assistant_content: list[dict[str, Any]] = []
                if text:
                    assistant_content.append({"type": "text", "text": text})
                for tu in tool_uses:
                    assistant_content.append({
                        "type": "tool_use", "id": tu.id,
                        "name": tu.name, "input": tu.input,
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                # Execute tools and collect results
                tool_results: list[dict[str, Any]] = []
                for tu in tool_uses:
                    result_msg = await self._execute_tool(tu.name, tu.input)
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": tu.id, "content": result_msg,
                    })
                    if tu.name == "recall_memo":
                        memo_reads += 1
                    elif tu.name == "update_memo":
                        memo_writes += 1
                    dream_logger.debug(
                        "tool {} | id={} | result={}",
                        tu.name, tu.input.get("id", "?"), result_msg[:80],
                    )
                messages.append({"role": "user", "content": tool_results})
            else:
                dream_logger.warning(
                    "dream exhausted max rounds | reads={} writes={} elapsed={:.1f}s",
                    memo_reads, memo_writes, time.time() - t0,
                )

            dream_logger.info("dream completed")
        except Exception:
            dream_logger.exception("dream failed")
        finally:
            self._running = False

    async def _execute_tool(self, name: str, inp: dict[str, Any]) -> str:
        """Execute a dream tool call and return the result string."""
        if name == "recall_memo":
            memo_id = inp.get("id", "")
            if not memo_id:
                return "缺少 id 参数"
            memo = self._store.read(memo_id)
            if memo is None:
                return f"未找到 memo: {memo_id}"
            return memo.body

        if name == "update_memo":
            memo_id = inp.get("id", "")
            memo_content = inp.get("memo", "")
            if not memo_id or not memo_content:
                return "缺少 id 或 memo 参数"
            try:
                await self._store.write(memo_id, memo_content, "dream")
                return "已更新"
            except (ValueError, OSError) as e:
                return f"写入失败: {e}"

        return f"未知工具: {name}"
