"""HTTP API 调用工具：通用 REST 接口调用。"""

import json
from typing import Any

import httpx

from src.tools.base import Tool
from src.tools.context import ToolContext
from src.tools.web_fetch import _is_safe_url

MAX_RESPONSE_LENGTH = 4000


class HttpApiTool(Tool):
    @property
    def name(self) -> str:
        return "http_api"

    @property
    def description(self) -> str:
        return "调用外部 HTTP API，支持 GET/POST。适合查询天气、翻译、汇率等第三方服务。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["GET", "POST"], "description": "HTTP 方法", "default": "GET"},
                "url": {"type": "string", "description": "API URL"},
                "headers": {"type": "object", "description": "请求头", "additionalProperties": {"type": "string"}},
                "body": {"type": "object", "description": "POST 请求体（JSON）"},
            },
            "required": ["url"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        url: str = kwargs["url"]
        if not _is_safe_url(url):
            return "拒绝访问: 不允许访问内网地址"

        method: str = kwargs.get("method", "GET")
        if method not in ("GET", "POST"):
            return f"不支持的 HTTP 方法: {method}"
        headers: dict[str, str] = kwargs.get("headers", {})
        body: dict[str, Any] | None = kwargs.get("body")

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            if method == "POST":
                resp = await client.post(url, headers=headers, json=body)
            else:
                resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        try:
            data = resp.json()
            text = json.dumps(data, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, ValueError):
            text = resp.text

        if len(text) > MAX_RESPONSE_LENGTH:
            text = text[:MAX_RESPONSE_LENGTH] + "...(已截断)"
        return text
