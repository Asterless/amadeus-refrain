"""LLM client: assemble message lists, call Anthropic API, handle tool loops."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
from loguru import logger

from src.identity.models import Identity
from src.llm.prompt import PromptBuilder
from src.llm.usage import UsageTracker
from src.memory.group_timeline import GroupTimeline
from src.memory.image_cache import ImageCache
from src.memory.memo_store import MemoStore
from src.memory.short_term import ChatMessage, ShortTermMemory
from src.memory.types import Content
from src.tools.context import ToolContext
from src.tools.registry import ToolRegistry

MAX_TOOL_ROUNDS = 5
_MAX_COMPACT_TOOL_ROUNDS = 3
_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_BASE_DELAY = 5.0  # seconds
_SEGMENT_SEP = "---cut---"
_SEGMENT_DELAY = 0.5  # seconds between segment sends
_BLANK_LINE_RE = re.compile(r"\n{2,}")
_CQ_CODE_RE = re.compile(r"\[CQ:[^\]]+\]")
_CQ_KV_FIX_RE = re.compile(r",(\w+):")


class RateLimitError(RuntimeError):
    """Raised when the Anthropic API returns a rate-limit error."""


def _clean_text(text: str) -> str:
    """Collapse consecutive blank lines into a single newline."""
    return _BLANK_LINE_RE.sub("\n", text).strip()


def _fix_cq_codes(text: str) -> str:
    """Normalize CQ code params: [CQ:reply,id:123] → [CQ:reply,id=123]."""
    return _CQ_CODE_RE.sub(lambda m: _CQ_KV_FIX_RE.sub(r",\1=", m.group(0)), text)


def _split_segments(text: str) -> list[str]:
    """Split reply into multiple messages by --- separator, cleaning blank lines."""
    text = _fix_cq_codes(text)
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


def _to_anthropic_message(msg: ChatMessage) -> dict[str, Any]:
    return {"role": msg["role"], "content": msg["content"]}


def _content_text(content: Content) -> str:
    """Extract plain text from Content, ignoring image blocks."""
    if isinstance(content, str):
        return content
    return " ".join(b["text"] for b in content if b["type"] == "text")


async def _resolve_image_refs(
    messages: list[dict[str, Any]],
    image_cache: ImageCache | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Convert image_ref blocks to Anthropic image blocks (base64).

    Returns (messages, image_tag_map) where image_tag_map maps "img:N" to disk paths.
    """
    image_tag_map: dict[str, str] = {}
    if image_cache is None:
        return messages, image_tag_map

    t0 = time.perf_counter()
    resolved_count = 0
    tag_counter = 0

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        # Collect async resolve tasks with their indices
        tasks: list[tuple[int, dict[str, Any], asyncio.Task[dict[str, Any] | None]]] = []
        for i, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "image_ref":
                task = asyncio.ensure_future(image_cache.load_as_base64(block))
                tasks.append((i, block, task))

        if not tasks:
            continue

        await asyncio.gather(*(t for _, _, t in tasks))

        new_content: list[dict[str, Any]] = []
        task_map = {i: (block, task) for i, block, task in tasks}
        for i, block in enumerate(content):
            if i not in task_map:
                new_content.append(block)
                continue
            orig_block, task = task_map[i]
            cache_ctrl = orig_block.get("cache_control")
            resolved = task.result()
            if resolved is not None:
                tag_counter += 1
                tag = f"img:{tag_counter}"
                image_tag_map[tag] = orig_block["path"]
                new_content.append({"type": "text", "text": f"«{tag}»"})
                if cache_ctrl:
                    resolved = {**resolved, "cache_control": cache_ctrl}
                new_content.append(resolved)
                resolved_count += 1
            else:
                fallback: dict[str, Any] = {"type": "text", "text": "«图片已过期»"}
                if cache_ctrl:
                    fallback["cache_control"] = cache_ctrl
                new_content.append(fallback)
        msg["content"] = new_content

    if resolved_count:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("resolve_image_refs | images={} tags={} elapsed={:.0f}ms", resolved_count, tag_counter, elapsed_ms)

    return messages, image_tag_map


def _hash_json(obj: Any) -> str:
    """Return first 8 hex chars of SHA-256 of JSON-serialized obj."""
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def _log_cache_debug(
    session_id: str,
    system_blocks: list[dict[str, Any]],
    tool_defs: list[dict[str, Any]],
    msg_count: int,
) -> None:
    tools_hash = _hash_json(tool_defs)
    block_hashes = [_hash_json(b) for b in system_blocks]
    logger.debug(
        "cache_debug | session={} tools={} system=[{}] msgs={}",
        session_id, tools_hash, ", ".join(block_hashes), msg_count,
    )


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

_COMPACT_MEMO_TOOL: dict[str, Any] = {
    "name": "append_memo",
    "description": "向用户或群组的备忘录追加新观察（自动写入「待整理」区域）。不要重复已有内容，只记新信息。",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "目标 memo ID，如 user_QQ号 或 group_群号",
            },
            "note": {
                "type": "string",
                "description": "要追加的新观察（一句话简短结论，系统自动加 bullet 前缀）",
            },
        },
        "required": ["id", "note"],
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
    """Call Anthropic API, parse SSE stream."""
    body: dict[str, Any] = {
        "model": model,
        "system": system_blocks,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        # Cache-control on last tool def so the whole tool set is cached together
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

    t_ser = time.perf_counter()
    payload_bytes = json.dumps(body).encode()
    payload = io.BytesIO(payload_bytes)
    logger.trace(
        "api payload serialized | size={}KB elapsed={:.1f}ms",
        len(payload_bytes) // 1024,
        (time.perf_counter() - t_ser) * 1000,
    )
    async with session.post(f"{base_url}/v1/messages", data=payload, headers=headers) as resp:
        if resp.status == 429:
            body_text = await resp.text()
            raise RateLimitError(f"HTTP 429: {body_text}")
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
            elif event_type == "message_delta":
                delta_usage: dict[str, Any] = data.get("usage", {})
                for k, v in delta_usage.items():
                    if isinstance(v, int):
                        usage[k] = v
            elif event_type == "error":
                error_data = data.get("error", {})
                error_msg = error_data.get("message", str(data))
                if "rate limit" in error_msg.lower():
                    raise RateLimitError(f"Anthropic API stream error: {error_msg}")
                raise RuntimeError(f"Anthropic API stream error: {error_msg}")

    # Token stats
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    input_tokens = usage.get("input_tokens", 0)
    total_input = input_tokens + cache_read + cache_create
    output_tokens = usage.get("output_tokens", 0)

    return {
        "text": "".join(text_parts),
        "tool_uses": tool_uses,
        "input_tokens": total_input,
        "output_tokens": output_tokens,
        "cache_read": cache_read,
        "cache_create": cache_create,
    }


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
        compress_ratio: float = 0.5,
        max_compact_failures: int = 3,
        group_timeline: GroupTimeline | None = None,
        memo_store: MemoStore | None = None,
        bot_self_id: str = "",
        on_compact: Callable[[], None] | None = None,
        image_cache: ImageCache | None = None,
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
        self._compress_ratio = compress_ratio
        self._max_compact_failures = max_compact_failures
        self._private_compact_failures: int = 0
        self._group_compact_failures: int = 0
        self._timeline = group_timeline
        self._memo_store = memo_store
        self._bot_self_id = bot_self_id
        self._on_compact = on_compact
        self._usage_tracker: UsageTracker | None = None
        self._image_cache = image_cache

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

    def _record_usage(
        self,
        *,
        call_type: str,
        user_id: str,
        group_id: str | None,
        input_tokens: int,
        cache_read_tokens: int,
        cache_create_tokens: int,
        output_tokens: int,
        tool_rounds: int,
        elapsed_s: float,
        error: str | None = None,
    ) -> None:
        """Fire-and-forget usage recording."""
        if not self._usage_tracker:
            return
        asyncio.create_task(self._usage_tracker.record(  # noqa: RUF006
            call_type=call_type,
            user_id=user_id or None,
            group_id=group_id,
            model=self._model,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_create_tokens=cache_create_tokens,
            output_tokens=output_tokens,
            tool_rounds=tool_rounds,
            elapsed_s=elapsed_s,
            error=error,
        ))

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_group_messages(self, group_id: str) -> list[dict[str, Any]]:
        """Build message list for group chat: optional summary + turns + pending + cache breakpoint."""
        assert self._timeline is not None
        messages: list[dict[str, Any]] = []

        # Summary as stable prefix for cache hits
        summary = self._timeline.get_summary(group_id)
        if summary:
            messages.append({
                "role": "user",
                "content": [_cached_text(f"«对话摘要»\n{summary}")],
            })
            messages.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

        # Turns — finalized, byte-identical to previous API calls
        messages.extend(self._timeline.get_turns(group_id))

        # Pending — temporary merge as tail user message
        pending = self._timeline.get_pending(group_id)
        if pending:
            from src.memory.group_timeline import _merge_user_contents
            messages.append({"role": "user", "content": _merge_user_contents(pending)})

        # Place cache breakpoint at the position recorded by the previous API call
        cached_idx = self._timeline.get_cached_msg_index(group_id)
        if 0 < cached_idx < len(messages):
            target = messages[cached_idx]
            content = target.get("content")
            if isinstance(content, str):
                messages[cached_idx] = {"role": target["role"], "content": [_cached_text(content)]}
            elif isinstance(content, list):
                content = [*content]
                content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
                messages[cached_idx] = {"role": target["role"], "content": content}

        # Store second-to-last for next call (last may grow with new pending)
        if len(messages) >= 2:
            self._timeline.set_cached_msg_index(group_id, len(messages) - 2)

        return messages

    def _build_private_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Build message list for private chat: optional summary + history + cache breakpoint."""
        messages: list[dict[str, Any]] = []

        # Summary as stable prefix
        summary = self._short_term.get_summary(session_id)
        if summary:
            messages.append({
                "role": "user",
                "content": [_cached_text(f"«对话摘要»\n{summary}")],
            })
            messages.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

        # Chat history
        history = self._short_term.get(session_id)
        for i, msg in enumerate(history):
            m = _to_anthropic_message(msg)
            if i == len(history) - 2:
                content = m["content"]
                if isinstance(content, str):
                    m = {"role": m["role"], "content": [_cached_text(content)]}
                elif isinstance(content, list):
                    content = [*content]
                    content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
                    m = {"role": m["role"], "content": content}
            messages.append(m)

        return messages

    # ------------------------------------------------------------------
    # Main chat entry point
    # ------------------------------------------------------------------

    async def chat(
        self,
        session_id: str,
        user_id: str,
        user_content: Content,
        identity: Identity,
        group_id: str | None = None,
        ctx: ToolContext | None = None,
        on_segment: Callable[[str], Awaitable[None]] | None = None,
    ) -> str | None:
        content_preview = user_content[:80] if isinstance(user_content, str) else str(user_content)[:80]
        logger.info(
            "chat | session={} user={} identity={} text={!r}",
            session_id, user_id, identity.id, content_preview,
        )
        t0 = time.monotonic()

        is_group = group_id is not None and self._timeline is not None

        if is_group:
            assert group_id is not None
            assert self._timeline is not None
            # Group: messages already added to timeline by group_listener
            if self._timeline.needs_compact(group_id, self._max_context_tokens, self._compact_ratio):
                logger.info(
                    "compact triggering | group={} input_tokens={} threshold={}",
                    group_id, self._timeline.get_input_tokens(group_id),
                    int(self._max_context_tokens * self._compact_ratio),
                )
                await self._compact_group(group_id, identity)
            messages = self._build_group_messages(group_id)
        else:
            # Private: use ShortTermMemory
            self._short_term.add(session_id, "user", user_content)
            if self._short_term.needs_compact(session_id, self._max_context_tokens, self._compact_ratio):
                logger.info(
                    "compact triggering | session={} input_tokens={} threshold={}",
                    session_id, self._short_term.get_input_tokens(session_id),
                    int(self._max_context_tokens * self._compact_ratio),
                )
                await self._compact(session_id)
            messages = self._build_private_messages(session_id)

        if self._memo_store:
            try:
                system_blocks = await self._prompt.build_blocks(
                    user_id=user_id,
                    group_id=group_id,
                    memo_store=self._memo_store,
                )
            except Exception:
                logger.exception("build_blocks failed, falling back to static block")
                system_blocks = [self._prompt.static_block]
        else:
            system_blocks = [self._prompt.static_block]

        messages, image_tag_map = await _resolve_image_refs(messages, self._image_cache)

        tool_defs: list[dict[str, Any]] | None = None
        if not self._tools.empty:
            tool_defs = _to_anthropic_tools(self._tools.to_openai_tools())
        # pass_turn is always available
        tool_defs = [*(tool_defs or []), _PASS_TURN_TOOL]

        # Debug: hash system blocks and tools to diagnose cache misses
        _log_cache_debug(session_id, system_blocks, tool_defs, len(messages))

        # Token accumulators across tool rounds
        acc_input = 0
        acc_output = 0
        acc_cache_read = 0
        acc_cache_create = 0

        for round_i in range(MAX_TOOL_ROUNDS):
            result = await self._call(system_blocks, messages, tools=tool_defs)
            acc_input += result["input_tokens"] - result.get("cache_read", 0) - result.get("cache_create", 0)
            acc_output += result.get("output_tokens", 0)
            acc_cache_read += result.get("cache_read", 0)
            acc_cache_create += result.get("cache_create", 0)
            text: str = result["text"]
            tool_uses: list[_ToolUse] = result["tool_uses"]

            # Check for pass_turn
            pass_turn = next((tu for tu in tool_uses if tu.name == "pass_turn"), None)
            if pass_turn:
                reason = pass_turn.input.get("reason", "")
                elapsed = time.monotonic() - t0
                logger.info("pass_turn | session={} reason={!r} elapsed={:.1f}s", session_id, reason, elapsed)
                if is_group and group_id is not None and self._timeline is not None:
                    self._timeline.set_input_tokens(group_id, result["input_tokens"])
                self._record_usage(
                    call_type="proactive",
                    user_id=user_id, group_id=group_id,
                    input_tokens=acc_input, cache_read_tokens=acc_cache_read,
                    cache_create_tokens=acc_cache_create, output_tokens=acc_output,
                    tool_rounds=round_i, elapsed_s=elapsed,
                )
                return None

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
                if is_group and group_id is not None and self._timeline is not None:
                    self._timeline.add(group_id, role="assistant", content=full_reply)
                    self._timeline.set_input_tokens(group_id, result["input_tokens"])
                else:
                    self._short_term.add(session_id, "assistant", full_reply)
                    self._short_term.set_input_tokens(session_id, result["input_tokens"])
                self._record_usage(
                    call_type="proactive" if is_group else "chat",
                    user_id=user_id, group_id=group_id,
                    input_tokens=acc_input, cache_read_tokens=acc_cache_read,
                    cache_create_tokens=acc_cache_create, output_tokens=acc_output,
                    tool_rounds=round_i, elapsed_s=elapsed,
                )
                return last_seg

            for tu in tool_uses:
                logger.info(
                    "tool_call | round={} name={} args={!r}",
                    round_i, tu.name, json.dumps(tu.input, ensure_ascii=False)[:200],
                )

            # Assistant message content
            assistant_content: list[dict[str, Any]] = []
            if text:
                assistant_content.append({"type": "text", "text": text})
            for tu in tool_uses:
                assistant_content.append({"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input})
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute tools in parallel
            tool_ctx = ctx or ToolContext(user_id=user_id, group_id=group_id)
            tool_ctx.extra["image_tags"] = image_tag_map
            if self._timeline is not None:
                tool_ctx.extra["timeline"] = self._timeline
            call_results = await asyncio.gather(
                *[self._tools.call(tu.name, json.dumps(tu.input), ctx=tool_ctx) for tu in tool_uses],
                return_exceptions=True,
            )
            # Convert any exceptions to error strings
            call_results = [
                r if isinstance(r, str) else f"Tool error: {r}" for r in call_results
            ]
            tool_results: list[dict[str, Any]] = []
            for tu, result_text in zip(tool_uses, call_results, strict=True):
                logger.debug("tool_result | name={} result={!r}", tu.name, result_text[:200])
                tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_text})
            messages.append({"role": "user", "content": tool_results})

        logger.warning("tool loop exhausted | session={} rounds={}", session_id, MAX_TOOL_ROUNDS)
        result = await self._call(system_blocks, messages)
        acc_input += result["input_tokens"] - result.get("cache_read", 0) - result.get("cache_create", 0)
        acc_output += result.get("output_tokens", 0)
        acc_cache_read += result.get("cache_read", 0)
        acc_cache_create += result.get("cache_create", 0)
        reply = result["text"] or "..."
        segments = _split_segments(reply)
        if on_segment and len(segments) > 1:
            for seg in segments[:-1]:
                await on_segment(seg)
                await asyncio.sleep(_SEGMENT_DELAY)
        last_seg = segments[-1]
        full_reply = "\n".join(segments)
        if is_group and group_id is not None and self._timeline is not None:
            self._timeline.add(group_id, role="assistant", content=full_reply)
            self._timeline.set_input_tokens(group_id, result["input_tokens"])
        else:
            self._short_term.add(session_id, "assistant", full_reply)
            self._short_term.set_input_tokens(session_id, result["input_tokens"])
        elapsed = time.monotonic() - t0
        self._record_usage(
            call_type="proactive" if is_group else "chat",
            user_id=user_id, group_id=group_id,
            input_tokens=acc_input, cache_read_tokens=acc_cache_read,
            cache_create_tokens=acc_cache_create, output_tokens=acc_output,
            tool_rounds=MAX_TOOL_ROUNDS, elapsed_s=elapsed,
        )
        return last_seg

    # ------------------------------------------------------------------
    # Compact — private chat
    # ------------------------------------------------------------------

    async def _compact_with_tools(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        source: str,
        group_id: str | None,
    ) -> tuple[str, int]:
        """Run a compact LLM call with an update_memo tool loop.

        Returns (summary_text, memo_writes).
        """
        tools: list[dict[str, Any]] | None = None
        if self._memo_store:
            tools = [_COMPACT_MEMO_TOOL]

        acc_input = 0
        acc_output = 0
        acc_cache_read = 0
        acc_cache_create = 0
        memo_writes = 0

        for round_i in range(_MAX_COMPACT_TOOL_ROUNDS):
            result = await _call_api(
                self._session, self._base_url, self._api_key, self._model,
                system, messages, max_tokens=1024, tools=tools,
            )
            acc_input += result["input_tokens"] - result.get("cache_read", 0) - result.get("cache_create", 0)
            acc_output += result.get("output_tokens", 0)
            acc_cache_read += result.get("cache_read", 0)
            acc_cache_create += result.get("cache_create", 0)

            text: str = result["text"].strip()
            tool_uses: list[_ToolUse] = result["tool_uses"]

            if not tool_uses:
                self._record_usage(
                    call_type="compact", user_id="", group_id=group_id,
                    input_tokens=acc_input, cache_read_tokens=acc_cache_read,
                    cache_create_tokens=acc_cache_create, output_tokens=acc_output,
                    tool_rounds=round_i, elapsed_s=0.0,
                )
                return text, memo_writes

            # Build assistant message with text + tool_use blocks
            assistant_content: list[dict[str, Any]] = []
            if text:
                assistant_content.append({"type": "text", "text": text})
            for tu in tool_uses:
                assistant_content.append({"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input})
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute append_memo tool calls
            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                if tu.name == "append_memo" and self._memo_store:
                    memo_id = tu.input.get("id", "")
                    note = tu.input.get("note", "")
                    if memo_id and note:
                        try:
                            await self._memo_store.append(memo_id, note, source)
                            memo_writes += 1
                            logger.debug("compact memo append | id={} source={}", memo_id, source)
                            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": "已追加"})
                        except (ValueError, OSError) as e:
                            logger.warning("compact memo append failed | id={} error={}", memo_id, e)
                            result_msg = f"写入失败: {e}"
                            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_msg})
                    else:
                        tool_results.append({
                            "type": "tool_result", "tool_use_id": tu.id, "content": "缺少 id 或 note 参数",
                        })
                else:
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": "未知工具"})
            messages.append({"role": "user", "content": tool_results})

        # Exhausted rounds — do a final call without tools to get summary
        result = await _call_api(
            self._session, self._base_url, self._api_key, self._model,
            system, messages, max_tokens=1024,
        )
        acc_input += result["input_tokens"] - result.get("cache_read", 0) - result.get("cache_create", 0)
        acc_output += result.get("output_tokens", 0)
        acc_cache_read += result.get("cache_read", 0)
        acc_cache_create += result.get("cache_create", 0)
        self._record_usage(
            call_type="compact", user_id="", group_id=group_id,
            input_tokens=acc_input, cache_read_tokens=acc_cache_read,
            cache_create_tokens=acc_cache_create, output_tokens=acc_output,
            tool_rounds=_MAX_COMPACT_TOOL_ROUNDS, elapsed_s=0.0,
        )
        return result["text"].strip(), memo_writes

    async def _compact(self, session_id: str) -> None:
        """Compress first half of history into summary and extract user memo."""
        if self._private_compact_failures >= self._max_compact_failures:
            history = self._short_term.get(session_id)
            drop = max(2, int(len(history) * self._compress_ratio))
            self._short_term.drop_oldest(session_id, drop)
            logger.warning("compact circuit breaker active, dropped {} msgs | session={}", drop, session_id)
            return

        try:
            history = self._short_term.get(session_id)
            if len(history) < 4:
                return

            old_summary = self._short_term.get_summary(session_id)
            split = max(2, int(len(history) * self._compress_ratio))

            # Assemble content for compression
            lines: list[str] = []
            if old_summary:
                lines.append(f"«之前的对话摘要»\n{old_summary}\n")
            for msg in history[:split]:
                role_label = "用户" if msg["role"] == "user" else "助手"
                lines.append(f"{role_label}: {_content_text(msg['content'])}")
            conversation_text = "\n".join(lines)

            # Extract user_id from session_id (format: "private_{user_id}")
            user_id = ""
            if self._memo_store and session_id.startswith("private_"):
                user_id = session_id[len("private_"):]

            if self._memo_store and user_id:
                system = [{"type": "text", "text": (
                    "你是一个对话压缩助手。请完成两个任务：\n"
                    "1. 将以下对话历史压缩成简洁的中文摘要。"
                    "保留关键信息：用户的问题、重要决策、关键结论、用户偏好。"
                    "去掉寒暄、重复内容和过程性细节。\n"
                    "2. 如果对话中出现了关于用户的新信息（性格、偏好、背景等），"
                    f"用 append_memo 工具追加到 user_{user_id}。"
                    "每条 note 写一句话结论，系统会自动放入「待整理」区域。"
                    "没有新信息则不需要调用。\n"
                    "备忘录规则：只记新的印象和结论，不记流水账。\n"
                    "最终请输出纯摘要文本（不要加标题或格式）。"
                )}]
            else:
                system = [{"type": "text", "text": (
                    "你是一个对话压缩助手。请将以下对话历史压缩成简洁的中文摘要。"
                    "保留关键信息：用户的问题、重要决策、关键结论、用户偏好。"
                    "去掉寒暄、重复内容和过程性细节。输出纯摘要文本，不要加标题或格式。"
                )}]
            compress_messages: list[dict[str, Any]] = [{"role": "user", "content": conversation_text}]

            logger.info("compact | session={} split={}/{}", session_id, split, len(history))
            source = f"compact:private:{session_id}"
            t_compact = time.monotonic()
            new_summary, memo_writes = await self._compact_with_tools(
                system, compress_messages, source, group_id=None,
            )
            compact_elapsed = time.monotonic() - t_compact

            if new_summary:
                self._short_term.compact(session_id, split, new_summary)
                logger.info(
                    "compact done | session={} messages={}->{} summary_len={} memo_writes={} elapsed={:.1f}s",
                    session_id, len(history), len(history) - split,
                    len(new_summary), memo_writes, compact_elapsed,
                )
            else:
                logger.warning("compact produced empty summary | session={}", session_id)

            # Rebuild system blocks so updated memos are reflected
            if memo_writes > 0 and user_id:
                self._prompt.invalidate(user_id=user_id)

            self._private_compact_failures = 0
            if self._on_compact:
                self._on_compact()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._private_compact_failures += 1
            logger.exception("compact failed ({}/{})", self._private_compact_failures, self._max_compact_failures)

    # ------------------------------------------------------------------
    # Compact — group chat (with memo extraction)
    # ------------------------------------------------------------------

    async def _compact_group(self, group_id: str, identity: Identity) -> None:
        """Compress first half of group timeline into summary and extract memos."""
        if self._group_compact_failures >= self._max_compact_failures:
            assert self._timeline is not None
            turns = self._timeline.get_turns(group_id)
            drop = max(2, int(len(turns) * self._compress_ratio))
            self._timeline.drop_oldest(group_id, drop)
            logger.warning("compact circuit breaker active, dropped {} turns | group={}", drop, group_id)
            return

        try:
            assert self._timeline is not None
            turns = self._timeline.get_turns(group_id)
            if len(turns) < 4:
                return

            old_summary = self._timeline.get_summary(group_id)
            split = max(2, int(len(turns) * self._compress_ratio))

            # Assemble content for compression
            # Note: turns don't carry speaker info (merged into content at flush time).
            # Task 6 will rewrite this to query SQLite for raw messages with speaker info.
            lines: list[str] = []
            if old_summary:
                lines.append(f"«之前的对话摘要»\n{old_summary}\n")
            for turn in turns[:split]:
                if turn["role"] == "assistant":
                    lines.append(f"{identity.name}: {_content_text(turn['content'])}")
                else:
                    lines.append(f"{_content_text(turn['content'])}")
            conversation_text = "\n".join(lines)

            # Collect user IDs from turn content (speaker info is embedded in merged content)
            seen_user_ids: list[str] = []
            if self._memo_store:
                seen: set[str] = set()
                for turn in turns[:split]:
                    if turn["role"] == "user":
                        text = _content_text(turn["content"])
                        for m in re.finditer(r"\((\d+)\)", text):
                            uid = m.group(1)
                            if uid not in seen:
                                seen.add(uid)
                                seen_user_ids.append(uid)

            system = [{"type": "text", "text": (
                "你是一个对话分析助手。请完成两个任务：\n"
                "1. 将以下群聊记录压缩成简洁的中文摘要。保留关键信息。\n"
                "2. 如果对话中出现了关于用户或群组的新信息，"
                "用 append_memo 工具追加新观察。每条 note 写一句话结论，"
                "系统会自动放入「待整理」区域。没有新信息则不需要调用。\n\n"
                "**关键规则——个人情报与群情报分离：**\n"
                "- 关于某个人的信息（性格、偏好、背景、身份、观点、与他人的关系）"
                "→ 写入该用户的备忘录（user_QQ号）\n"
                "- 只有群级别信息（群氛围、群事件、群规矩、成员变动）"
                "→ 写入群备忘录（group_群号）\n"
                "- 判断标准：如果信息跟着这个人走（换个群也成立），写 user_；"
                "如果只在本群语境下有意义，写 group_\n\n"
                f"本群 ID: group_{group_id}\n"
                f"出现的用户 ID: {', '.join(f'user_{uid}' for uid in seen_user_ids)}\n\n"
                "备忘录规则：用 @QQ号 标注人物，#群号 标注群。QQ号是唯一身份标识，昵称不可信。\n"
                "只记新的印象和结论，不记流水账。\n"
                "最终请输出纯摘要文本（不要加标题或格式）。"
            )}]
            compress_messages: list[dict[str, Any]] = [{"role": "user", "content": conversation_text}]

            logger.info("compact_group | group={} split={}/{}", group_id, split, len(turns))
            source = f"compact:group:{group_id}"
            t_compact = time.monotonic()
            new_summary, memo_writes = await self._compact_with_tools(
                system, compress_messages, source, group_id=group_id,
            )
            compact_elapsed = time.monotonic() - t_compact

            if new_summary:
                self._timeline.compact(group_id, split, new_summary)
                logger.info(
                    "compact_group done | group={} turns={}->{} summary_len={} memo_writes={} elapsed={:.1f}s",
                    group_id, len(turns), len(turns) - split,
                    len(new_summary), memo_writes, compact_elapsed,
                )
            else:
                logger.warning("compact_group produced empty summary | group={}", group_id)

            # Rebuild system blocks so updated memos are reflected
            if memo_writes > 0:
                self._prompt.invalidate(group_id=group_id)

            self._group_compact_failures = 0
            if self._on_compact:
                self._on_compact()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._group_compact_failures += 1
            logger.exception("compact_group failed ({}/{})", self._group_compact_failures, self._max_compact_failures)


