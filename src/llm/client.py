"""LLM 客户端封装：拼装消息列表，调用 Anthropic API，处理工具调用。"""

import json
from typing import Any

import aiohttp
from loguru import logger

from src.identity.models import Identity
from src.llm.prompt import PromptBuilder
from src.memory.short_term import ChatMessage, ShortTermMemory
from src.tools.context import ToolContext
from src.tools.registry import ToolRegistry

MAX_TOOL_ROUNDS = 5


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
        "anthropic-version": "2023-06-01",
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

    return {"text": "".join(text_parts), "tool_uses": tool_uses}


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt_builder: PromptBuilder,
        short_term: ShortTermMemory,
        tools: ToolRegistry,
    ) -> None:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._prompt = prompt_builder
        self._short_term = short_term
        self._tools = tools

    async def _call(
        self, system_blocks: list[dict[str, Any]], messages: list[Any], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        return await _call_api(
            self._session, self._base_url, self._api_key, self._model, system_blocks, messages, tools=tools
        )

    async def chat(
        self,
        session_id: str,
        user_id: str,
        user_text: str,
        identity: Identity,
        group_id: str | None = None,
        ctx: ToolContext | None = None,
    ) -> str:
        logger.info("chat | session={} user={} identity={} text={!r}", session_id, user_id, identity.id, user_text[:80])
        self._short_term.add(session_id, "user", user_text)

        system_blocks = await self._prompt.build_blocks(identity=identity, user_id=user_id, group_id=group_id)

        messages: list[Any] = []

        # 群聊记录 → messages 开头，带 cache_control
        if group_id:
            ctx_text = self._prompt.build_context_message(group_id)
            if ctx_text:
                messages.append({
                    "role": "user",
                    "content": [_cached_text(f"[群聊上下文]\n{ctx_text}")],
                })
                messages.append({"role": "assistant", "content": "好的，我已了解最近的群聊内容。"})

        # 对话历史：倒数第二条加 cache（之前的对话不会变）
        history = self._short_term.get(session_id)
        for i, msg in enumerate(history):
            m = _to_anthropic_message(msg)
            if i == len(history) - 2:
                m = {"role": m["role"], "content": [_cached_text(m["content"])]}
            messages.append(m)

        tool_defs: list[dict[str, Any]] | None = None
        if not self._tools.empty:
            tool_defs = _to_anthropic_tools(self._tools.to_openai_tools())

        for round_i in range(MAX_TOOL_ROUNDS):
            result = await self._call(system_blocks, messages, tools=tool_defs)
            text: str = result["text"]
            tool_uses: list[_ToolUse] = result["tool_uses"]

            if not tool_uses:
                reply = text or "..."
                logger.info("reply | session={} len={}", session_id, len(reply))
                self._short_term.add(session_id, "assistant", reply)
                return reply

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
        self._short_term.add(session_id, "assistant", reply)
        return reply
