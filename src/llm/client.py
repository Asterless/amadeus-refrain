"""LLM 客户端封装：拼装消息列表，调用 Anthropic API，处理工具调用。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
from loguru import logger

from src.identity.models import Identity
from src.llm.prompt import PromptBuilder
from src.memory.group_timeline import GroupTimeline
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ChatMessage, ShortTermMemory
from src.tools.context import ToolContext
from src.tools.registry import ToolRegistry

MAX_TOOL_ROUNDS = 5
_SEGMENT_SEP = "---"
_SEGMENT_DELAY = 0.5  # 分段发送间隔（秒）
_BLANK_LINE_RE = __import__("re").compile(r"\n{2,}")


def _clean_text(text: str) -> str:
    """压缩连续空行为单个换行。"""
    return _BLANK_LINE_RE.sub("\n", text).strip()


def _split_segments(text: str) -> list[str]:
    """按 --- 分隔符拆分回复为多条消息，并清理空行。"""
    segments: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip() == _SEGMENT_SEP:
            seg = "\n".join(current).strip()
            if seg:
                segments.append(_clean_text(seg))
            current = []
        else:
            current.append(line)
    last = "\n".join(current).strip()
    if last:
        segments.append(_clean_text(last))
    return segments or [_clean_text(text)]


class _ToolUse:
    __slots__ = ("id", "input", "name")

    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:
        self.id = id
        self.name = name
        self.input = input


def _cached_text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


def _to_anthropic_message(msg: ChatMessage) -> dict[str, str]:
    return {"role": msg["role"], "content": msg["content"]}


def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"]["parameters"],
        }
        for t in tools
    ]


_PASS_TURN_TOOL: dict[str, Any] = {
    "name": "pass_turn",
    "description": "当你认为不需要回复时调用此工具，跳过本轮发言。",
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "不回复的简短原因",
            }
        },
        "required": ["reason"],
    },
}


async def _call_api(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    system_blocks: list[dict[str, Any]],
    messages: list[Any],
    max_tokens: int = 1024,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """直接调用 Anthropic API，解析 SSE 流。"""
    body: dict[str, Any] = {
        "model": model,
        "system": system_blocks,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        # 工具定义最后一个加 cache_control，整组工具一起缓存
        cached_tools = [*tools]
        cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}
        body["tools"] = cached_tools

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2024-10-22",
    }

    text_parts: list[str] = []
    tool_uses: list[_ToolUse] = []
    current_tool: dict[str, str] = {}
    usage: dict[str, int] = {}

    async with session.post(f"{base_url}/v1/messages", json=body, headers=headers) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data: dict[str, Any] = json.loads(line[6:])
            event_type = data.get("type", "")

            if event_type == "message_start":
                msg_usage: dict[str, Any] = data.get("message", {}).get("usage", {})
                usage = {k: v for k, v in msg_usage.items() if isinstance(v, int)}
            elif event_type == "content_block_start":
                block: dict[str, Any] = data.get("content_block", {})
                if block.get("type") == "tool_use":
                    current_tool = {"id": block["id"], "name": block["name"], "input_json": ""}
            elif event_type == "content_block_delta":
                delta: dict[str, Any] = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_parts.append(delta["text"])
                elif delta.get("type") == "input_json_delta":
                    current_tool["input_json"] += delta.get("partial_json", "")
            elif event_type == "content_block_stop":
                if current_tool:
                    input_data: dict[str, Any] = (
                        json.loads(current_tool["input_json"]) if current_tool["input_json"] else {}
                    )
                    tool_uses.append(_ToolUse(id=current_tool["id"], name=current_tool["name"], input=input_data))
                    current_tool = {}

    # 记录 cache 命中情况
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    input_tokens = usage.get("input_tokens", 0)
    if cache_read or cache_create:
        total = input_tokens + cache_read + cache_create
        hit_rate = (cache_read / total * 100) if total else 0
        logger.info(
            "cache | input={} cache_read={} cache_create={} hit={:.0f}%",
            input_tokens, cache_read, cache_create, hit_rate,
        )

    total_input = input_tokens + cache_read + cache_create
    return {"text": "".join(text_parts), "tool_uses": tool_uses, "input_tokens": total_input}


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt_builder: PromptBuilder,
        short_term: ShortTermMemory,
        tools: ToolRegistry,
        max_context_tokens: int = 200_000,
        compact_ratio: float = 0.7,
        group_timeline: GroupTimeline | None = None,
        long_term: LongTermMemory | None = None,
        bot_self_id: str = "",
    ) -> None:
        connector = aiohttp.TCPConnector(
            enable_cleanup_closed=True,
            keepalive_timeout=15,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=120, sock_read=30),
        )
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._prompt = prompt_builder
        self._short_term = short_term
        self._tools = tools
        self._max_context_tokens = max_context_tokens
        self._compact_ratio = compact_ratio
        self._timeline = group_timeline
        self._long_term = long_term
        self._bot_self_id = bot_self_id

    async def close(self) -> None:
        await self._session.close()

    async def _call(
        self, system_blocks: list[dict[str, Any]], messages: list[Any], tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        return await _call_api(
            self._session, self._base_url, self._api_key, self._model,
            system_blocks, messages, max_tokens=max_tokens, tools=tools,
        )

    # ------------------------------------------------------------------
    # 消息构建
    # ------------------------------------------------------------------

    def _build_group_messages(self, group_id: str) -> list[dict[str, Any]]:
        """为群聊构建 messages 列表：可选摘要 + 时间线消息 + 缓存断点。"""
        assert self._timeline is not None
        messages: list[dict[str, Any]] = []

        # 摘要 → 稳定前缀
        summary = self._timeline.get_summary(group_id)
        if summary:
            messages.append({
                "role": "user",
                "content": [_cached_text(f"[对话摘要]\n{summary}")],
            })
            messages.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

        # 时间线消息（不加缓存断点：每次消息列表都变，断点漂移反而打破 cache；
        # 稳定的缓存断点只放在 system blocks、tools 和上面的摘要上）
        history = self._timeline.to_anthropic_messages(group_id)
        messages.extend(history)

        return messages

    def _build_private_messages(self, session_id: str) -> list[dict[str, Any]]:
        """为私聊构建 messages 列表：可选摘要 + 对话历史 + 缓存断点。"""
        messages: list[dict[str, Any]] = []

        # 摘要 → 稳定前缀
        summary = self._short_term.get_summary(session_id)
        if summary:
            messages.append({
                "role": "user",
                "content": [_cached_text(f"[对话摘要]\n{summary}")],
            })
            messages.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

        # 对话历史
        history = self._short_term.get(session_id)
        for i, msg in enumerate(history):
            m = _to_anthropic_message(msg)
            if i == len(history) - 2:
                m = {"role": m["role"], "content": [_cached_text(m["content"])]}
            messages.append(m)

        return messages

    # ------------------------------------------------------------------
    # 主对话入口
    # ------------------------------------------------------------------

    async def chat(
        self,
        session_id: str,
        user_id: str,
        user_text: str,
        identity: Identity,
        group_id: str | None = None,
        ctx: ToolContext | None = None,
        on_segment: Callable[[str], Awaitable[None]] | None = None,
        allow_skip: bool = False,
    ) -> str | None:
        logger.info("chat | session={} user={} identity={} text={!r}", session_id, user_id, identity.id, user_text[:80])
        t0 = time.monotonic()

        is_group = group_id is not None and self._timeline is not None

        if is_group:
            # 群聊：消息已由 group_listener 添加到 timeline，不再 add
            if self._timeline.needs_compact(group_id, self._max_context_tokens, self._compact_ratio):
                await self._compact_group(group_id, identity)
            messages = self._build_group_messages(group_id)
        else:
            # 私聊：保持原有 ShortTermMemory 逻辑
            self._short_term.add(session_id, "user", user_text)
            if self._short_term.needs_compact(session_id, self._max_context_tokens, self._compact_ratio):
                await self._compact(session_id)
            messages = self._build_private_messages(session_id)

        system_blocks = await self._prompt.build_blocks(
            identity=identity, user_id=user_id, group_id=group_id, bot_self_id=self._bot_self_id,
        )

        tool_defs: list[dict[str, Any]] | None = None
        if not self._tools.empty:
            tool_defs = _to_anthropic_tools(self._tools.to_openai_tools())
        # Always include pass_turn so tool list is stable across @bot / scheduler calls (cache)
        tool_defs = [*(tool_defs or []), _PASS_TURN_TOOL]

        for round_i in range(MAX_TOOL_ROUNDS):
            result = await self._call(system_blocks, messages, tools=tool_defs)
            text: str = result["text"]
            tool_uses: list[_ToolUse] = result["tool_uses"]

            # Check for pass_turn (only honored when allow_skip=True)
            pass_turn = next((tu for tu in tool_uses if tu.name == "pass_turn"), None)
            if pass_turn and allow_skip:
                reason = pass_turn.input.get("reason", "")
                elapsed = time.monotonic() - t0
                logger.info("pass_turn | session={} reason={!r} elapsed={:.1f}s", session_id, reason, elapsed)
                if is_group:
                    self._timeline.set_input_tokens(group_id, result["input_tokens"])
                return None
            # Filter out pass_turn so it doesn't enter the tool-execution loop
            if pass_turn:
                tool_uses = [tu for tu in tool_uses if tu.name != "pass_turn"]

            if not tool_uses:
                reply = text or "..."
                segments = _split_segments(reply)
                if on_segment and len(segments) > 1:
                    for seg in segments[:-1]:
                        await on_segment(seg)
                        await asyncio.sleep(_SEGMENT_DELAY)
                last_seg = segments[-1]
                full_reply = "\n".join(segments)
                elapsed = time.monotonic() - t0
                logger.info(
                    "reply | session={} len={} segments={} elapsed={:.1f}s",
                    session_id, len(full_reply), len(segments), elapsed,
                )
                if is_group:
                    self._timeline.add(group_id, role="assistant", content=full_reply)
                    self._timeline.set_input_tokens(group_id, result["input_tokens"])
                else:
                    self._short_term.add(session_id, "assistant", full_reply)
                    self._short_term.set_input_tokens(session_id, result["input_tokens"])
                return last_seg

            for tu in tool_uses:
                logger.info(
                    "tool_call | round={} name={} args={!r}",
                    round_i, tu.name, json.dumps(tu.input, ensure_ascii=False)[:200],
                )

            # assistant 消息
            assistant_content: list[dict[str, Any]] = []
            if text:
                assistant_content.append({"type": "text", "text": text})
            for tu in tool_uses:
                assistant_content.append({"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input})
            messages.append({"role": "assistant", "content": assistant_content})

            # 执行工具
            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                tool_ctx = ctx or ToolContext(user_id=user_id, group_id=group_id)
                tool_result = await self._tools.call(tu.name, json.dumps(tu.input), ctx=tool_ctx)
                logger.debug("tool_result | name={} result={!r}", tu.name, tool_result[:200])
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": tool_result})
            messages.append({"role": "user", "content": tool_results})

        logger.warning("tool loop exhausted | session={} rounds={}", session_id, MAX_TOOL_ROUNDS)
        result = await self._call(system_blocks, messages)
        reply = result["text"] or "..."
        segments = _split_segments(reply)
        if on_segment and len(segments) > 1:
            for seg in segments[:-1]:
                await on_segment(seg)
                await asyncio.sleep(_SEGMENT_DELAY)
        last_seg = segments[-1]
        full_reply = "\n".join(segments)
        if is_group:
            self._timeline.add(group_id, role="assistant", content=full_reply)
            self._timeline.set_input_tokens(group_id, result["input_tokens"])
        else:
            self._short_term.add(session_id, "assistant", full_reply)
            self._short_term.set_input_tokens(session_id, result["input_tokens"])
        return last_seg

    # ------------------------------------------------------------------
    # Compact — 私聊
    # ------------------------------------------------------------------

    async def _compact(self, session_id: str) -> None:
        """将历史前半部分压缩成摘要。"""
        history = self._short_term.get(session_id)
        if len(history) < 4:
            return  # 消息太少，不值得 compact

        old_summary = self._short_term.get_summary(session_id)
        split = len(history) // 2

        # 拼装要压缩的内容
        lines: list[str] = []
        if old_summary:
            lines.append(f"[之前的对话摘要]\n{old_summary}\n")
        for msg in history[:split]:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}: {msg['content']}")
        conversation_text = "\n".join(lines)

        system = [{"type": "text", "text": (
            "你是一个对话压缩助手。请将以下对话历史压缩成简洁的中文摘要。"
            "保留关键信息：用户的问题、重要决策、关键结论、用户偏好。"
            "去掉寒暄、重复内容和过程性细节。输出纯摘要文本，不要加标题或格式。"
        )}]
        compress_messages = [{"role": "user", "content": conversation_text}]

        logger.info("compact | session={} split={}/{}", session_id, split, len(history))
        result = await _call_api(
            self._session, self._base_url, self._api_key, self._model,
            system, compress_messages, max_tokens=1024,
        )
        new_summary = result["text"].strip()
        if new_summary:
            self._short_term.compact(session_id, split, new_summary)
            logger.info("compact done | session={} summary_len={}", session_id, len(new_summary))
        else:
            logger.warning("compact produced empty summary | session={}", session_id)

    # ------------------------------------------------------------------
    # Compact — 群聊（增强：提取长期记忆）
    # ------------------------------------------------------------------

    async def _compact_group(self, group_id: str, identity: Identity) -> None:
        """将群聊时间线前半部分压缩成摘要，并提取用户长期记忆。"""
        assert self._timeline is not None
        messages = self._timeline.get_messages(group_id)
        if len(messages) < 4:
            return

        old_summary = self._timeline.get_summary(group_id)
        split = len(messages) // 2

        # 拼装要压缩的内容
        lines: list[str] = []
        if old_summary:
            lines.append(f"[之前的对话摘要]\n{old_summary}\n")
        for msg in messages[:split]:
            if msg["role"] == "assistant":
                lines.append(f"{identity.name}: {msg['content']}")
            elif msg["speaker"]:
                lines.append(f"{msg['speaker']}: {msg['content']}")
            else:
                lines.append(f"用户: {msg['content']}")
        conversation_text = "\n".join(lines)

        system = [{"type": "text", "text": (
            "你是一个对话分析助手。请完成两个任务：\n"
            "1. 将以下群聊记录压缩成简洁的中文摘要。保留关键信息：讨论话题、重要决策、关键结论。\n"
            "2. 提取值得长期记住的用户信息（昵称、爱好、特征、关键事件）。\n\n"
            "以 JSON 格式输出：\n"
            '{"summary": "摘要文本", "memories": [{"user_id": "QQ号", "nickname": "昵称", '
            '"traits": {"key": "value"}, "event": "事件描述"}]}\n\n'
            "如果没有值得记住的用户信息，memories 为空数组。只输出 JSON，不要加其他文字。"
        )}]
        compress_messages = [{"role": "user", "content": conversation_text}]

        logger.info("compact_group | group={} split={}/{}", group_id, split, len(messages))
        result = await _call_api(
            self._session, self._base_url, self._api_key, self._model,
            system, compress_messages, max_tokens=1024,
        )
        raw_text = result["text"].strip()

        new_summary = ""
        memories: list[dict[str, Any]] = []
        try:
            parsed = json.loads(raw_text)
            new_summary = parsed.get("summary", "")
            memories = parsed.get("memories", [])
        except (json.JSONDecodeError, TypeError):
            logger.warning("compact_group JSON parse failed, using raw text as summary | group={}", group_id)
            new_summary = raw_text

        if new_summary:
            self._timeline.compact(group_id, split, new_summary)
            logger.info(
                "compact_group done | group={} summary_len={} memories={}",
                group_id, len(new_summary), len(memories),
            )
        else:
            logger.warning("compact_group produced empty summary | group={}", group_id)

        # 写入长期记忆
        if self._long_term and memories:
            for mem in memories:
                uid = mem.get("user_id", "")
                if not uid:
                    continue
                nickname = mem.get("nickname")
                traits = mem.get("traits")
                event = mem.get("event")
                if nickname or traits:
                    await self._long_term.update_profile(uid, nickname=nickname, traits=traits)
                if event:
                    await self._long_term.add_event(uid, event)

