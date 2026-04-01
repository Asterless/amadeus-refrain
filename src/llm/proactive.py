"""主动插话评估器：用廉价模型判断是否应主动加入群聊。

消息到达后攒 batch，等消息间隙（debounce）或攒满后统一评估。
同一群同一时间只跑一个评估，评估期间的新消息标记为 pending，
评估结束后如有 pending 则立即再评估一轮。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

import aiohttp
from loguru import logger
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from src.identity.models import Identity
from src.memory.group_timeline import GroupTimeline


class ProactiveDecision(TypedDict):
    """单条插话决策。"""

    reason: str  # 插话原因
    reply_to: str  # 回复对象（昵称或 "大家"）


class _GroupBatch:
    """单个群的 batch 状态。"""

    __slots__ = ("debounce_task", "evaluating", "msg_count", "pending")

    def __init__(self) -> None:
        self.msg_count: int = 0  # 自上次评估后的消息计数
        self.debounce_task: asyncio.Task[None] | None = None  # 当前 debounce 定时器
        self.evaluating: bool = False  # 是否正在评估
        self.pending: bool = False  # 评估期间是否有新消息待处理


class ProactiveEvaluator:
    """消息批量评估器：debounce + batch size 触发，单并发 + pending 队列。"""

    def __init__(
        self,
        *,
        timeline: GroupTimeline,
        model: str,
        api_key: str,
        base_url: str,
        enabled: bool = True,
        timeout: float = 15.0,
        context_lines: int = 20,
        cooldown: int = 60,
        batch_timeout: float = 5.0,
        batch_size: int = 10,
    ) -> None:
        self._enabled = enabled
        self._timeline = timeline
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._context_lines = context_lines
        self._cooldown = cooldown
        self._batch_timeout = batch_timeout
        self._batch_size = batch_size

        self._batches: dict[str, _GroupBatch] = {}
        self._last_proactive: dict[str, float] = {}
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def notify(
        self,
        group_id: str,
        identity: Identity,
        on_reply: Callable[[list[ProactiveDecision]], Awaitable[None]],
    ) -> None:
        """通知有新群消息。内部处理 debounce、batch、评估和回调。"""
        if not self._enabled or not identity.proactive:
            return

        # 冷却期内不处理
        last = self._last_proactive.get(group_id, 0.0)
        if time.monotonic() - last < self._cooldown:
            return

        batch = self._batches.setdefault(group_id, _GroupBatch())
        batch.msg_count += 1

        # 如果正在评估，标记 pending 然后返回
        if batch.evaluating:
            batch.pending = True
            logger.debug("batch | group={} pending (evaluating) msgs={}", group_id, batch.msg_count)
            return

        # 取消旧的 debounce 定时器
        if batch.debounce_task and not batch.debounce_task.done():
            batch.debounce_task.cancel()
            logger.debug("batch | group={} debounce reset msgs={}", group_id, batch.msg_count)

        # 达到 batch_size → 立即触发
        if batch.msg_count >= self._batch_size:
            logger.info("batch | group={} full ({} msgs) → evaluating", group_id, batch.msg_count)
            self._fire(group_id, identity, on_reply)
        else:
            # 启动 debounce 定时器
            logger.debug(
                "batch | group={} debounce start msgs={} wait={:.1f}s",
                group_id, batch.msg_count, self._batch_timeout,
            )
            batch.debounce_task = asyncio.create_task(
                self._debounce(group_id, identity, on_reply),
            )

    async def close(self) -> None:
        # 取消所有 debounce 定时器
        for batch in self._batches.values():
            if batch.debounce_task and not batch.debounce_task.done():
                batch.debounce_task.cancel()
        if self._session is not None:
            await self._session.close()

    # ------------------------------------------------------------------
    # 内部：触发与 debounce
    # ------------------------------------------------------------------

    async def _debounce(
        self,
        group_id: str,
        identity: Identity,
        on_reply: Callable[[list[ProactiveDecision]], Awaitable[None]],
    ) -> None:
        """等待消息间隙后触发评估。"""
        try:
            await asyncio.sleep(self._batch_timeout)
            batch = self._batches.get(group_id)
            msgs = batch.msg_count if batch else 0
            logger.info("batch | group={} debounce fired ({} msgs) → evaluating", group_id, msgs)
            self._fire(group_id, identity, on_reply)
        except asyncio.CancelledError:
            pass  # 被新消息取消了，正常

    def _fire(
        self,
        group_id: str,
        identity: Identity,
        on_reply: Callable[[list[ProactiveDecision]], Awaitable[None]],
    ) -> None:
        """启动评估后台任务。"""
        batch = self._batches.get(group_id)
        if not batch:
            return
        batch.msg_count = 0
        batch.pending = False
        batch.evaluating = True

        task = asyncio.create_task(self._eval_loop(group_id, identity, on_reply))
        task.add_done_callback(lambda _: None)

    async def _eval_loop(
        self,
        group_id: str,
        identity: Identity,
        on_reply: Callable[[list[ProactiveDecision]], Awaitable[None]],
    ) -> None:
        """评估循环：评估完毕后如有 pending 则再评估一轮。"""
        batch = self._batches.get(group_id)
        if not batch:
            return

        round_num = 0
        try:
            while True:
                round_num += 1
                logger.info("batch | group={} eval round={}", group_id, round_num)
                decisions = await self._evaluate(group_id, identity)
                if decisions:
                    self._last_proactive[group_id] = time.monotonic()
                    for d in decisions:
                        logger.info(
                            "batch | group={} decided to reply | reply_to={!r} reason={!r}",
                            group_id, d["reply_to"], d["reason"],
                        )
                    try:
                        await on_reply(decisions)
                    except Exception:
                        logger.exception("proactive reply error | group={}", group_id)
                else:
                    logger.info("batch | group={} decided not to reply", group_id)

                # 检查是否有 pending
                if batch.pending and not decisions:
                    # 只在上一轮没决定插话时才续评（插话后进冷却期）
                    logger.info("batch | group={} has pending msgs → re-evaluating", group_id)
                    batch.pending = False
                    batch.msg_count = 0
                    continue
                break
        finally:
            batch.evaluating = False
            batch.pending = False
            batch.msg_count = 0
            logger.info("batch | group={} eval done rounds={}", group_id, round_num)

    # ------------------------------------------------------------------
    # 决策 prompt 构建
    # ------------------------------------------------------------------

    DECISION_FORMAT = (
        "用 JSON 回答，不要输出其他内容。\n"
        "想插话时列出回复对象（可多个）：\n"
        '{"replies": [{"reason": "简短原因", "reply_to": "回复对象昵称或大家"}, ...]}\n'
        "不想插话时返回空数组：\n"
        '{"replies": []}'
    )

    def build_decision_prompt(
        self, group_id: str, identity: Identity,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """构建决策用的 system blocks 和 messages。"""
        messages = self._timeline.get_messages(group_id)
        recent = messages[-self._context_lines :]

        lines: list[str] = []
        for msg in recent:
            if msg["role"] == "assistant":
                lines.append(f"{identity.name}: {msg['content']}")
            elif msg["speaker"]:
                lines.append(f"{msg['speaker']}: {msg['content']}")
            else:
                lines.append(msg["content"])

        system_text = f"{identity.personality}\n\n{identity.proactive}\n\n{self.DECISION_FORMAT}"
        system = [{"type": "text", "text": system_text}]
        user_messages = [{"role": "user", "content": "\n".join(lines)}]
        return system, user_messages

    # ------------------------------------------------------------------
    # 决策调用
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    async def _evaluate(self, group_id: str, identity: Identity) -> list[ProactiveDecision]:
        """调用廉价模型判断是否应插话。失败最多重试 3 次。"""
        try:
            text = await self._call_decision_api(group_id, identity)
            logger.info("proactive eval | group={} raw={!r}", group_id, text[:100])
            return self._parse_decision(text)
        except Exception:
            logger.warning("proactive eval failed after retries | group={}", group_id, exc_info=True)
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
    async def _call_decision_api(self, group_id: str, identity: Identity) -> str:
        """调用决策 API，带 tenacity 重试。"""
        system, messages = self.build_decision_prompt(group_id, identity)

        body = {
            "model": self._model,
            "system": system,
            "messages": messages,
            "max_tokens": 128,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2024-10-22",
        }

        async with self._get_session().post(
            f"{self._base_url}/v1/messages", json=body, headers=headers,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            return text.strip()

    @staticmethod
    def _parse_decision(text: str) -> list[ProactiveDecision]:
        """解析决策模型的 JSON 输出。支持批量和单条格式，容错处理。"""
        # 提取 JSON
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start:end])
                except json.JSONDecodeError:
                    return []
            else:
                return []

        # 新格式：{"replies": [...]}
        if "replies" in parsed:
            results: list[ProactiveDecision] = []
            for item in parsed["replies"]:
                if isinstance(item, dict):
                    results.append(ProactiveDecision(
                        reason=str(item.get("reason", "")),
                        reply_to=str(item.get("reply_to", "")),
                    ))
            return results

        # 兼容旧格式：{"reply": true, "reason": "...", "reply_to": "..."}
        if not parsed.get("reply"):
            return []
        return [ProactiveDecision(
            reason=str(parsed.get("reason", "")),
            reply_to=str(parsed.get("reply_to", "")),
        )]
