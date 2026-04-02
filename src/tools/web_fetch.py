"""网页抓取工具：获取 URL 内容。"""

import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from src.tools.base import Tool
from src.tools.context import ToolContext

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
MAX_LENGTH = 4000


def _is_safe_url(url: str) -> bool:
    """拒绝内网/本机地址，防止 SSRF。"""
    import socket
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if not hostname:
            return False
        if hostname in ("localhost", "host.docker.internal", "napcat"):
            return False
        # 解析域名，检查所有 IP 是否为公网地址
        try:
            addrinfos = socket.getaddrinfo(hostname, None)
            for _, _, _, _, sockaddr in addrinfos:
                addr = ipaddress.ip_address(sockaddr[0])
                if not _is_allowed_addr(addr):
                    return False
            return bool(addrinfos)
        except socket.gaierror:
            return False
    except Exception:
        return False


# 198.18.0.0/15: 本地 DNS 代理（如 Clash）将公网域名解析到此段
_PROXY_DNS_NET = ipaddress.ip_network("198.18.0.0/15")


def _is_allowed_addr(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """判断地址是否允许访问：全局地址或代理 DNS 段。"""
    if addr.is_global:
        return True
    return isinstance(addr, ipaddress.IPv4Address) and addr in _PROXY_DNS_NET


class WebFetchTool(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "抓取指定 URL 的网页内容，返回纯文本。适合查询在线信息、文档、新闻等。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要抓取的网页 URL"},
            },
            "required": ["url"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        url: str = kwargs["url"]
        if not _is_safe_url(url):
            return "拒绝访问: 不允许访问内网地址"

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 QQBot/1.0"})
            resp.raise_for_status()

        text = _SCRIPT_STYLE_RE.sub(" ", resp.text)
        text = _TAG_RE.sub(" ", text)
        text = _SPACE_RE.sub(" ", text).strip()

        if len(text) > MAX_LENGTH:
            text = text[:MAX_LENGTH] + "...(已截断)"
        return text
