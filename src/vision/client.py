"""Free vision preprocessing: describe images via an OpenAI-compatible vision API.

The main chat LLM (e.g. DeepSeek) may be text-only — it silently drops image
blocks. Instead of sending raw images to it, we first ask a free vision model
(e.g. Zhipu GLM-4.6V-Flash) to describe the image, then feed the description
as plain text to the chat model. Descriptions are cached in-process so each
image is described at most once per process lifetime.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

_DESCRIBE_PROMPT = (
    "请用中文描述这张图片：画面里有什么、人物的表情/动作、整体情绪或梗、适合什么聊天场景。"
    "控制在80字以内，直接输出描述，不要加任何前缀。"
)

_STICKER_PROMPT = (
    "你是表情包审核员。仔细看这张表情包，判断画面内容和情绪/梗，"
    "用中文输出严格 JSON（不要 markdown、不要多余文字）：\n"
    '{"description": "画面内容和情绪/梗，20字以内", '
    '"usage_hint": "最贴切的使用场景，20字以内，要具体不要套话'
    '（例如：群友说离谱消息时、馋了想要什么时、被夸了得意时）"}'
)


def _parse_sticker_json(text: str) -> tuple[str, str] | None:
    """Parse {description, usage_hint} from the model output."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    desc = str(data.get("description", "")).strip()
    hint = str(data.get("usage_hint", "")).strip()
    if not desc or not hint:
        return None
    return desc, hint


class VisionClient:
    """Call an OpenAI-compatible /chat/completions endpoint to describe images."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 300,
        timeout_s: float = 30.0,
        max_concurrency: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._session: aiohttp.ClientSession | None = None
        self._desc_cache: dict[str, str] = {}
        self._sticker_cache: dict[str, tuple[str, str]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _post_image(self, prompt: str, image_path: str, media_type: str) -> str | None:
        """Send an image with the given prompt to the vision API, return text or None."""
        path = Path(image_path)
        if not path.exists():
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        b64 = base64.b64encode(data).decode("ascii")

        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        t0 = time.perf_counter()
        async with self._sem:
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        resp_text = await resp.text()
                        logger.warning(
                            "vision describe failed | model={} status={} resp={}",
                            self._model, resp.status, resp_text[:200],
                        )
                        return None
                    payload: dict[str, Any] = await resp.json()
            except Exception:
                logger.warning(
                    "vision describe error | model={} path={}",
                    self._model, image_path,
                    exc_info=True,
                )
                return None

        try:
            desc = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.warning("vision describe bad response | model={}", self._model)
            return None

        desc = (desc or "").strip()
        if not desc:
            return None

        self._desc_cache[image_path] = desc
        logger.debug(
            "vision describe ok | model={} path={} chars={} elapsed={:.0f}ms",
            self._model, image_path, len(desc), (time.perf_counter() - t0) * 1000,
        )
        return desc

    async def describe(self, image_path: str, media_type: str = "image/jpeg") -> str | None:
        """Return a short text description of the image, or None on failure."""
        cached = self._desc_cache.get(image_path)
        if cached is not None:
            return cached
        text = await self._post_image(_DESCRIBE_PROMPT, image_path, media_type)
        if text is None:
            return None
        self._desc_cache[image_path] = text
        return text

    async def describe_sticker(
        self, image_path: str, media_type: str = "image/jpeg"
    ) -> tuple[str, str] | None:
        """Return (description, usage_hint) for a sticker image, or None on failure."""
        cached = self._sticker_cache.get(image_path)
        if cached is not None:
            return cached
        text = await self._post_image(_STICKER_PROMPT, image_path, media_type)
        if text is None:
            return None
        pair = _parse_sticker_json(text)
        if pair is None:
            logger.warning(
                "vision sticker bad json | model={} resp={!r}",
                self._model, text[:200],
            )
            return None
        self._sticker_cache[image_path] = pair
        return pair
