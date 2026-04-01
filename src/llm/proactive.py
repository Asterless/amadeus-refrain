"""主动插话评估器：用廉价模型判断是否应主动加入群聊。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, TypedDict

import aiohttp
from loguru import logger

from src.identity.models import Identity
from src.memory.group_timeline import GroupTimeline


class ProactiveDecision(TypedDict):
    """决策结果。"""

    reason: str  # 插话原因
    reply_to: str  # 回复对象（昵称或 "大家"）


class ProactiveEvaluator:
    """评估是否应主动插话，并在决定插话时调用回调。"""

    def __init__(
        self,
        *,
        timeline: GroupTimeline,
        model: str,
        api_key: str,
        base_url: str,
        enabled: bool = True,
        timeout: float = 3.0,
        context_lines: int = 20,
        cooldown: int = 60,
    ) -> None:
        self._enabled = enabled
        self._timeline = timeline
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._context_lines = context_lines
        self._cooldown = cooldown

        self._locks: dict[str, asyncio.Lock] = {}
        self._last_proactive: dict[str, float] = {}
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # 前置检查
    # ------------------------------------------------------------------

    def should_evaluate(self, group_id: str, identity: Identity) -> bool:
        """快速判断是否需要进行决策调用。"""
        if not self._enabled:
            return False
        if not identity.proactive:
            return False

        # 冷却期内不评估
        last = self._last_proactive.get(group_id, 0.0)
        if time.monotonic() - last < self._cooldown:
            return False

        # 已有评估/回复在进行中
        lock = self._locks.get(group_id)
        return not (lock and lock.locked())

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    # ------------------------------------------------------------------
    # 决策 prompt 构建
    # ------------------------------------------------------------------

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

        system = [{"type": "text", "text": f"{identity.personality}\n\n{identity.proactive}"}]
        user_messages = [{"role": "user", "content": "\n".join(lines)}]
        return system, user_messages

    # ------------------------------------------------------------------
    # 决策调用
    # ------------------------------------------------------------------

    async def evaluate(self, group_id: str, identity: Identity) -> ProactiveDecision | None:
        """调用廉价模型判断是否应插话。返回决策详情或 None（不插话）。"""
        lock = self._locks.setdefault(group_id, asyncio.Lock())

        async with lock:
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

            try:
                async with self._get_session().post(
                    f"{self._base_url}/v1/messages", json=body, headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    text = ""
                    for block in data.get("content", []):
                        if block.get("type") == "text":
                            text += block.get("text", "")
                    text = text.strip()
                    logger.info("proactive eval | group={} raw={!r}", group_id, text[:100])

                    decision = self._parse_decision(text)
                    if decision:
                        self._last_proactive[group_id] = time.monotonic()
                        logger.info(
                            "proactive eval | group={} reply_to={!r} reason={!r}",
                            group_id, decision["reply_to"], decision["reason"],
                        )
                    return decision
            except TimeoutError:
                logger.warning("proactive eval timeout | group={}", group_id)
                return None
            except Exception:
                logger.warning("proactive eval error | group={}", group_id, exc_info=True)
                return None

    @staticmethod
    def _parse_decision(text: str) -> ProactiveDecision | None:
        """解析决策模型的 JSON 输出。容错处理各种格式。"""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start:end])
                except json.JSONDecodeError:
                    return None
            else:
                return None

        if not parsed.get("reply"):
            return None

        return ProactiveDecision(
            reason=str(parsed.get("reason", "")),
            reply_to=str(parsed.get("reply_to", "")),
        )

    def record_proactive(self, group_id: str) -> None:
        """手动记录插话时间（用于外部调用成功后）。"""
        self._last_proactive[group_id] = time.monotonic()
